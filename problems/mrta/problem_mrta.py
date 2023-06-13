from torch.utils.data import Dataset
import torch
import os
import pickle

from problems.mrta.state_mrta import StateMRTA
from utils.beam_search import beam_search


class MRTA(object):
    NAME = 'mrta'  # Capacitated Vehicle Routing Problem

    # VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size, loc_vec_size = dataset['loc'].size()
        # print(batch_size, graph_size, loc_vec_size)
        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        assert (
                       torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
                       sorted_pi[:, -graph_size:]
               ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))
        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        cost = (
                       (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
                       + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
                       + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
               ), None

        return cost

    @staticmethod
    def make_dataset(*args, **kwargs):
        return MRTADataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateMRTA.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = MRTA.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


def make_instance(data):
    loc = data['loc']
    workload = data['workload']
    deadline = data['deadline']
    initial_size = data["loc"].shape[0]
    n_agents = len(data['robots_work_capacity'])
    max_capacity = 10
    max_speed = .02
    grid_size = 1

    return [{
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'deadline': torch.tensor(deadline, dtype=torch.float),
        'depot': torch.zeros((1, 2)),
        'workload': torch.tensor(workload, dtype=torch.float),
        'initial_size': initial_size,
        'n_agents': n_agents,
        'max_n_agents': torch.tensor([[n_agents]]),
        'max_capacity': max_capacity,
        'max_speed': max_speed,
        'robots_start_location': torch.tensor(data['robots_start_location'], dtype=torch.float),
        'robots_work_capacity': torch.tensor(data['robots_work_capacity'], dtype=torch.float)

    }]


class MRTADataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0,
                 n_depot=1,
                 initial_size=None,
                 deadline_min=None,
                 deadline_max=None,
                 n_agents=20,
                 max_speed=.1,
                 distribution=None):
        super(MRTADataset, self).__init__()

        self.data_set = []
        if filename is not None:
            # assert os.path.splitext(filename)[1] == '.pkl'
            #
            # with open(filename, 'rb') as f:
            #     data = pickle.load(f)
            self.data = make_instance(filename)  # [make_instance(args) for args in data[offset:offset+num_samples]]

        else:

            max_n_agent = 10

            n_agents_available = torch.tensor([2, 3, 5, 7])

            agents_ids = torch.randint(0, 4, (num_samples, 1))

            groups = torch.randint(1, 3, (num_samples, 1))

            dist = torch.randint(1, 5, (num_samples, 1))

            data = []

            for i in range(num_samples):
                n_agents = n_agents_available[agents_ids[i, 0].item()].item()
                agents_location = (torch.randint(0, 101, (max_n_agent, 2)).to(torch.float) / 100)

                loc = torch.FloatTensor(size, 2).uniform_(0, 1)
                workload = torch.FloatTensor(size).uniform_(.2, .2)
                d_low = (((loc[:, None, :].expand((size, max_n_agent, 2)) - agents_location[None].expand(
                    (size, max_n_agent, 2))).norm(2, -1).max() / max_speed) + 20).to(torch.int64) + 1
                d_high = ((35) * (45) * 100 / (380) + d_low).to(torch.int64) + 1
                d_low = d_low * (.5 * groups[i, 0])
                d_high = ((d_high * (.5 * groups[i, 0]) / 10).to(torch.int64) + 1) * 10
                deadline_normal = (torch.rand(size, 1) * (d_high - d_low) + d_low).to(torch.int64) + 1

                n_norm_tasks = dist[i, 0] * 25
                rand_mat = torch.rand(size, 1)
                k = n_norm_tasks.item()  # For the general case change 0.25 to the percentage you need
                k_th_quant = torch.topk(rand_mat.T, k, largest=False)[0][:, -1:]
                bool_tensor = rand_mat <= k_th_quant
                normal_dist_tasks = torch.where(bool_tensor, torch.tensor(1), torch.tensor(0))

                slack_tasks = (normal_dist_tasks - 1).to(torch.bool).to(torch.int64)

                normal_dist_tasks_deadline = normal_dist_tasks * deadline_normal

                slack_tasks_deadline = slack_tasks * d_high

                deadline_final = normal_dist_tasks_deadline + slack_tasks_deadline

                robots_start_location = (torch.randint(0, 101, (max_n_agent, 2)).to(torch.float) / 100).to(
                    device=deadline_final.device)

                robots_work_capacity = torch.randint(1, 3, (max_n_agent, 1), dtype=torch.float,
                                                     device=deadline_final.device).view(-1) / 100

                case_info = {
                    'loc': loc,
                    'depot': torch.FloatTensor(1, 2).uniform_(0, 1),
                    'deadline': deadline_final.to(torch.float).view(-1),
                    'workload': workload,
                    'initial_size': 100,
                    'n_agents': torch.tensor([[n_agents]]),
                    'max_n_agents': torch.tensor([[max_n_agent]]),
                    'max_speed': max_speed,
                    'robots_start_location': robots_start_location,
                    'robots_work_capacity': robots_work_capacity
                }

                data.append(case_info)

            self.data = data

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]