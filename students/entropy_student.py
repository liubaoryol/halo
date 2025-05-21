import dataclasses
import numpy as np


from .base import CuriousPupil

@dataclasses.dataclass
class Latent_entropy_based(CuriousPupil):
    student_type: str = 'intent_entropy'
    single_query_only: bool = False

    def query_oracle(self, oracle, num_queries=1):
        """Randomly selects a trajectory and a timestep
        Logs query,, changes annotated options
        returns changed trajectories
        """
        if self.annotated_options is None:
            self.annotated_options = np.ones(oracle.true_options.shape)
            self.annotated_options[:] = None
        
        if not getattr(self.dataset, 'latent_probs', False):
            print("latent probs not estimated before hand.")
            probs = self.dataset.get_probs(self)
        else:
            probs = self.dataset.latent_probs
        entropies = [self._entropy(prob) for prob in probs]
        # Get the #num_queries highest entropies
        shape = oracle.true_options.shape
        entropies_arr = np.zeros(shape)
        for traj_num, entr in enumerate(entropies):
            entropies_arr[traj_num][:len(entr)] = entr
        argss = np.argpartition(
            entropies_arr.reshape(-1), -num_queries)[-num_queries:]
        argss = entropies_arr.reshape(-1)[argss]
        idxs = np.in1d(entropies_arr, argss).reshape(entropies_arr.shape)

        # book keeping
        traj_nums, idx_queries = np.where(idxs)
        changed_trjs = set()
        for traj_num, idx_query in zip(traj_nums, idx_queries):
            self.log_query(traj_num, idx_query)
            self.annotated_options[traj_num, idx_query] = oracle.query(
            traj_num, idx_query)
            changed_trjs.add(traj_num)
        return changed_trjs
        
    def _entropy(self, array):
        return np.nansum(-array * np.log(array), 1)
