import torch
from typing import NamedTuple
import numpy as np
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateMRTA(NamedTuple):
    # Fixed input
    coords: torch.Tensor  # Depot + loc (coordinates of all the locations including the depot)
    distance_matrix: torch.Tensor # distance matrix for all the coordinates
    time_matrix: torch.Tensor # time matrix between all the coordinates using the speed of the agents
    deadline: torch.Tensor # deadline for all the tasks (special case for the depot, keep a very large time)
    workload: torch.Tensor
    tasks_finish_time: torch.Tensor
    # demand: torch.Tensor # we do not need this, we can remove this or set the demand as 1 or something equal to the quantity for 1 time delivery

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and demands tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows (this is basically the ids for all the location, which are considered as integers)



    # robot specific
    robots_initial_decision_sequence: torch.Tensor # for timestep 1, all robots need decision, so we set a sequence for this
    # robots_total_tasks_done # optional
    robots_next_decision_time: torch.Tensor # tracks the next decision time for all the robots
    robots_current_destination: torch.Tensor
    robots_start_point: torch.Tensor
    robots_work_capacity: torch.Tensor
    robots_start_location: torch.Tensor
    robots_current_destination_location: torch.Tensor


    #general - frequent changing variable
    current_time: torch.Tensor # stores the value for the current time
    robot_taking_decision: torch.Tensor  # stores the id of the robot which will take the next decision
    next_decision_time: torch.Tensor # time at which the next decision is made. (0 t begin with)
    previous_decision_time: torch.Tensor # time at which the previous decision was made


    # for performance tracking
    tasks_done_success: torch.Tensor # keeps track of all the task id which was done successfully
    tasks_visited: torch.Tensor # keeps track of all the tasks which are visited (successful or not)
    depot: torch.Tensor



    # State
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step


    n_agents : torch.Tensor
    max_speed : torch.Tensor
    n_nodes: torch.Tensor
    n_depot: torch.Tensor



    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_[:,:,self.n_depot:]
        else:
            return mask_long2bool(self.visited_, n=self.demand.size(-1))

    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):  # If tensor, idx all tensors by this tensor:
            return self._replace(
                ids=self.ids[key],
                prev_a=self.prev_a[key],
                visited_=self.visited_[key],
                lengths=self.lengths[key],
                cur_coord=self.cur_coord[key],
            )
        return super(StateMRTA, self).__getitem__(key)


    @staticmethod
    def initialize(input,
                   visited_dtype = torch.uint8):
        depot = input['depot']
        loc = input['loc']
        max_speed = input['max_speed'][0].item()
        coords = torch.cat((depot[:, :], loc), -2).to(device=loc.device)
        distance_matrix = (coords[:, :, None, :] - coords[:, None, :, :]).norm(p=2, dim=-1).to(device=loc.device)
        time_matrix = torch.mul(distance_matrix, (1/max_speed)).to(device=loc.device)
        deadline = input['deadline']
        workload = input['workload']
        n_agents = input['n_agents'].reshape(-1)[:, None]
        max_n_agent = input['max_n_agents'][0, 0, 0].item()
        max_speed = input['max_speed'][0].item()
        n_depot = input['depot'].size()[1]
        batch_size, n_loc, _ = loc.size()
        robots_initial_decision_sequence = torch.from_numpy(np.arange(0, max_n_agent)).expand((batch_size, max_n_agent)).to(device=loc.device)
        robots_start_location = input['robots_start_location']


        return StateMRTA(
            coords=coords,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, n_loc + 1,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=input['depot'][:, None, :],  # Add step dimension
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),  # Vector with length num_steps
            robots_initial_decision_sequence = robots_initial_decision_sequence,
            robots_next_decision_time =  ((robots_initial_decision_sequence > (n_agents - 1)).to(torch.float) * 10000).to(device=loc.device),
            current_time = torch.zeros((batch_size, 1), dtype=torch.float, device=loc.device),
            robot_taking_decision = torch.zeros((batch_size, 1), dtype=torch.int64, device=loc.device),
            next_decision_time = torch.zeros((batch_size, 1), dtype=torch.float, device=loc.device),
            previous_decision_time = torch.zeros((batch_size, 1), dtype=torch.float, device=loc.device),
            tasks_done_success = torch.zeros((batch_size, 1), dtype=torch.int64, device=loc.device),
            tasks_visited = torch.zeros((batch_size, 1), dtype=torch.int64, device=loc.device),
            distance_matrix = distance_matrix,
            time_matrix = time_matrix,
            deadline = deadline,
            tasks_finish_time = torch.zeros((batch_size, n_loc), dtype=torch.float, device=loc.device),
            workload=workload,
            robots_current_destination = torch.zeros((batch_size, max_n_agent), dtype=torch.int64, device=loc.device),
            robots_start_point = torch.zeros((batch_size, max_n_agent), dtype=torch.int64, device=loc.device),
            robots_work_capacity= input['robots_work_capacity'],
            robots_start_location = robots_start_location,
            robots_current_destination_location = robots_start_location,
            depot = torch.zeros((batch_size, 1), dtype=torch.int64, device=loc.device),
            n_agents = n_agents,
            n_nodes = input['loc'].size()[1],
            n_depot=n_depot,
            max_speed = max_speed,
        )

    def get_final_cost(self):

        assert self.all_finished()

        len = self.lengths + (self.coords[self.ids, 0, :] - self.cur_coord).norm(p=2, dim=-1)
        return len

    def update(self, selected):
        selected = selected[:, None]  # Add dimension for step
        prev_a = selected

        previous_time = self.current_time

        current_time = self.next_decision_time


        cur_coords = self.robots_current_destination_location[self.ids, self.robot_taking_decision]

        time = (cur_coords - self.coords[self.ids, selected]).norm(2,2)/self.max_speed
        worktime = torch.div(self.workload[self.ids.view(-1), selected.view(-1) - 1],
                             self.robots_work_capacity[
                                 self.ids.view(-1), self.robot_taking_decision[self.ids].view(-1)])
        self.robots_next_decision_time[self.ids, self.robot_taking_decision] += torch.add(time,worktime[:, None])


        non_zero_indices = torch.nonzero(selected)
        # print(non_zero_indices.size()[0])
        if non_zero_indices.size()[0] > 0:
            deadlines = self.deadline[self.ids.view(-1), selected.view(-1) - 1]
            dest_time = self.robots_next_decision_time[self.ids.view(-1), self.robot_taking_decision[self.ids].view(-1)]
            self.tasks_finish_time[self.ids, selected - 1] = dest_time[:, None]

            feas_ids = (deadlines > dest_time).nonzero()
            combined = torch.cat((non_zero_indices[:,0], feas_ids[:,0]))
            uniques, counts = combined.unique(return_counts=True)
            intersection = uniques[counts > 1]
            if intersection.size()[0] > 0:
                self.tasks_done_success[intersection] += 1
            self.tasks_visited[non_zero_indices[:, 0]] += 1

        self.robots_start_point[self.ids, self.robot_taking_decision] = self.robots_current_destination[
            self.ids, self.robot_taking_decision]
        self.robots_current_destination[self.ids, self.robot_taking_decision] = selected

        sorted_time, indices = torch.sort(self.robots_next_decision_time)

        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            # This works, will not set anything if prev_a -1 == -1 (depot)
            visited_ = mask_long_scatter(self.visited_, prev_a - 1)

        new_cur_coord = self.coords[self.ids, selected]
        self.robots_current_destination_location[self.ids, self.robot_taking_decision] = new_cur_coord
        lengths = self.lengths #+ (new_cur_coord - cur_coords).norm(p=2, dim=-1)
        visited_[:,:,0] = 0

        return self._replace(
            prev_a=prev_a, previous_decision_time = previous_time, current_time = current_time,
            robot_taking_decision = indices[self.ids,0],
            next_decision_time = sorted_time[self.ids,0],
            visited_=visited_,
            lengths=lengths, cur_coord=new_cur_coord,
            i=self.i + 1
        )

    def all_finished(self):
        # return self.i.item() >= self.demand.size(-1) and self.visited.all()
        return self.visited.all()

    def get_finished(self):
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_current_node(self):
        return self.robots_current_destination[self.ids, self.robot_taking_decision] #self.prev ## this has been changed

    def get_mask(self):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """
        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, 1:]
        else:
            visited_loc = mask_long2bool(self.visited_)

        mask_loc = visited_loc.to(torch.bool)  # | exceeds_cap

        # robot_taking_decision = self.robot_taking_decision

        # Cannot visit the depot if just visited and still unserved nodes
        mask_depot = torch.tensor(torch.ones((mask_loc.size()[0], 1)).clone().detach(), dtype=torch.bool, device=mask_loc.device) #(self.robots_current_destination[self.ids, robot_taking_decision] == 0) & ((mask_loc == 0).int().sum(-1) > 0)
        full_mask = torch.cat((mask_depot[:, :, None], mask_loc), -1)

        return full_mask

    def construct_solutions(self, actions):
        return actions