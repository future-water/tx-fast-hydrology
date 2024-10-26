import datetime
import asyncio
import logging
import multiprocessing as mp
import numpy as np
import pandas as pd
from tx_fast_hydrology.callbacks import BaseCallback

PROCESSES = mp.cpu_count() - 1

logger = logging.getLogger(__name__)

class ModelCollection():
    def __init__(self, model_dict, outer_startnodes, outer_endnodes, startnode_indices, endnode_indices):
        self.outer_startnodes = outer_startnodes
        self.outer_endnodes = outer_endnodes
        self.startnode_indices = startnode_indices
        self.endnode_indices = endnode_indices
        self.outer_indegree = np.bincount(outer_endnodes, minlength=outer_endnodes.size)
        self._indegree = self.outer_indegree.copy()
        self.models = {index : model_dict[index]['model'] for index in model_dict}
        self.terminal_nodes = {index : model_dict[index]['terminal_node'] for index in model_dict}
        self.entry_nodes = {index : model_dict[index]['entry_node'] for index in model_dict}
        self.node_map = {index : model_dict[index]['node_map'] for index in model_dict}
        self.inputs = {index : model_dict[index]['input'] for index in model_dict}

    @property
    def info(self):
        model_info = {
            'outer_startnodes' : self.outer_startnodes,
            'outer_endnodes' : self.outer_endnodes,
            'startnode_indices' : self.startnode_indices,
            'endnode_indices' : self.startnode_indices,
            'outer_indegree' : self.outer_indegree,
            'models' : self.models,
            'terminal_nodes' : self.terminal_nodes,
            'entry_nodes' : self.entry_nodes,
            'node_map' : self.node_map
        }
        return model_info

    def load_states(self):
        for key in self.models:
            self.models[key].load_state()

    def save_states(self):
        for key in self.models:
            self.models[key].save_state()


class Simulation():
    def __init__(self, model_collection):
        self.model_collection = model_collection
        self.outer_startnodes = model_collection.outer_startnodes
        self.outer_endnodes = model_collection.outer_endnodes
        self.outer_indegree = model_collection.outer_indegree
        self._indegree = model_collection._indegree
        self.models = model_collection.models
        self.terminal_nodes = model_collection.terminal_nodes
        self.entry_nodes = model_collection.entry_nodes
        self.node_map = model_collection.node_map
        self.inputs = model_collection.inputs
        self.outputs = {}

    def simulate(self):
        raise NotImplementedError
        outer_startnodes = self.outer_startnodes
        outer_endnodes = self.outer_endnodes
        outer_indegree = self.outer_indegree
        multi_outputs = {}
        m = outer_startnodes.size
        outer_indegree_t = outer_indegree.copy()
        outer_indegree_t[outer_startnodes == outer_endnodes] -= 1

        for k in range(m):
            startnode = outer_startnodes[k]
            endnode = outer_endnodes[k]
            while (outer_indegree_t[startnode] == 0):
                model_start = self.models[startnode]
                p_t_start = self.inputs[startnode]
                outputs = {}
                outputs[model_start.datetime] = model_start.o_t_next
                for state in model_start.simulate_iter(p_t_start, inc_t=True):
                    o_t_next = state.o_t_next
                    outputs[state.datetime] = o_t_next
                outputs = pd.DataFrame.from_dict(outputs, orient='index')
                outputs.index = pd.to_datetime(outputs.index, utc=True)
                outputs.columns = p_t_start.columns
                multi_outputs[startnode] = outputs
                print(startnode)
                if startnode != endnode:
                    model_end = self.models[endnode]
                    terminal_node = self.terminal_nodes[startnode]
                    entry_node = self.entry_nodes[startnode]
                    index_out = self.node_map[startnode][terminal_node]
                    index_in = self.node_map[endnode][entry_node]
                    reach_id_out = model_start.reach_ids[index_out]
                    reach_id_in = model_end.reach_ids[index_in]
                    i_t_prev = outputs[reach_id_out].shift(1).iloc[1:].fillna(0.)
                    i_t_next = outputs[reach_id_out].iloc[1:]
                    gamma_in = model_end.gamma[index_in]
                    alpha_in = model_end.alpha[index_in]
                    beta_in = model_end.beta[index_in]
                    self.inputs[endnode].loc[:, reach_id_in] += (alpha_in
                                                                 * i_t_next.values
                                                                 / gamma_in
                                                               + beta_in
                                                                 * i_t_prev.values
                                                                 / gamma_in)
                    outer_indegree_t[endnode] -= 1
                    startnode = endnode
                    endnode = outer_endnodes[startnode]
                else:
                    break
        return multi_outputs
    
    def load_states(self):
        self.model_collection.load_states()

    def save_states(self):
        self.model_collection.save_states()

class AsyncSimulation(Simulation):
    def __init__(self, model_collection):
        return super().__init__(model_collection)
    
    async def simulate(self):
        indegree = self.outer_indegree.copy()
        indegree[self.outer_endnodes == self.outer_startnodes] -= 1
        self._indegree = indegree
        try:
            asyncio.get_running_loop()
            loop_running = True
        except RuntimeError:
            loop_running = False
        if loop_running:
            await self._main()
        else:
            asyncio.run(self._main())
        return self.outputs

    async def _main(self):
        indegree = self._indegree
        async with asyncio.TaskGroup() as taskgroup:
            for index, predecessors in enumerate(indegree):
                if predecessors == 0:
                    model = self.models[index]
                    inputs = self.inputs[index]
                    taskgroup.create_task(self._simulate(taskgroup, model,
                                                         inputs, index))

    async def _simulate(self, taskgroup, model, inputs, index):
        logger.debug(f'Started job for sub-watershed {index}')
        start_time = model.datetime
        outputs = {}
        outputs[start_time] = model.o_t_next
        for state in model.simulate_iter(inputs, inc_t=True):
            current_time = state.datetime
            o_t_next = state.o_t_next
            outputs[current_time] = o_t_next
        outputs = pd.DataFrame.from_dict(outputs, orient='index')
        outputs.index = pd.to_datetime(outputs.index, utc=True)
        outputs.columns = inputs.columns
        self.outputs[index] = outputs
        taskgroup.create_task(self._accumulate(taskgroup, outputs, index))

    async def _accumulate(self, taskgroup, outputs, index):
        indegree = self._indegree
        startnode = index
        endnode = self.outer_endnodes[startnode]
        if startnode != endnode:
            model_start = self.models[startnode]
            model_end = self.models[endnode]
            inputs = self.inputs[endnode]
            terminal_node = self.terminal_nodes[startnode]
            entry_node = self.entry_nodes[startnode]
            index_out = self.node_map[startnode][terminal_node]
            index_in = self.node_map[endnode][entry_node]
            reach_id_out = model_start.reach_ids[index_out]
            reach_id_in = model_end.reach_ids[index_in]
            i_t_prev = outputs[reach_id_out].shift(1).iloc[1:].fillna(0.)
            i_t_next = outputs[reach_id_out].iloc[1:]
            gamma_in = model_end.gamma[index_in]
            alpha_in = model_end.alpha[index_in]
            beta_in = model_end.beta[index_in]
            inputs.loc[:, reach_id_in] += (alpha_in * i_t_next.values / gamma_in
                                        + beta_in * i_t_prev.values / gamma_in)
            indegree[endnode] -= 1
            if (indegree[endnode] == 0):
                taskgroup.create_task(self._simulate(taskgroup, model_end,
                                                     inputs, endnode))
        logger.debug(f'Finished job for sub-watershed {index}')


class CheckPoint(BaseCallback):
    def __init__(self, model, checkpoint_time=None, timedelta=None):
        self.model = model
        if checkpoint_time is None:
            if timedelta is None:
                raise ValueError('Either `checkpoint_time` or `timedelta` must not be `None`.')
            else:
                checkpoint_time = model.datetime + datetime.timedelta(seconds=timedelta)
        self.checkpoint_time = checkpoint_time
        self.timedelta = timedelta
        self.model_saved = False

    def __on_simulation_start__(self):
        timedelta = self.timedelta
        if timedelta is None:
            return None
        else:
            checkpoint_time = self.model.datetime + datetime.timedelta(seconds=timedelta)
            self.set_checkpoint(checkpoint_time)
    
    def __on_step_end__(self):
        checkpoint_time = self.checkpoint_time
        current_time = self.model.datetime
        if (current_time >= checkpoint_time) and (not self.model_saved):
            self.model.save_state()
            self.model_saved = True

    def __on_save_state__(self):
        model = self.model
        for _, callback in model.callbacks.items():
            if hasattr(callback, 'save_state'):
                callback.save_state()

    def __on_load_state__(self):
        model = self.model
        for _, callback in model.callbacks.items():
            if hasattr(callback, 'load_state'):
                callback.load_state()
    
    def set_checkpoint(self, checkpoint_time):
        logger.info(f'Setting checkpoint time to {checkpoint_time}')
        self.checkpoint_time = checkpoint_time
        self.model_saved = False