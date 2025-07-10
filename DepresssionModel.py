#%%
# from init_rcParams import set_mpl_settings
# set_mpl_settings()
#%%
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
# import network_from_pd as FHS_net TODO: replace with your own network import

from dynsimf.models.Model import Model
from dynsimf.models.Model import ModelConfiguration

from dynsimf.models.components.PropertyFunction import PropertyFunction

class DepressionModel:
    

    def __init__(self, g, alpha_i_mp=1, alpha_r_mp=1, beta_i_mp=1, beta_r_mp=1, mtp_a_i_individual=np.array([1, 1, 1]), 
                 mtp_a_r_individual=np.array([1, 1, 1]), mtp_b_i_individual=np.array([1, 1, 1, 1]), mtp_b_r_individual=np.array([1, 1, 1]), 
                 flat_ai=0, flat_ar=0, flat_bi=0, flat_br=0, number_of_iterations=200):
        
        self.g = g
        self.model = Model(self.g)
        
        self.alpha_i_mp = alpha_i_mp
        self.alpha_r_mp = alpha_r_mp
        self.beta_i_mp = beta_i_mp
        self.beta_r_mp = beta_r_mp
        self.mtp_a_i_individual = mtp_a_i_individual
        self.mtp_a_r_individual = mtp_a_r_individual
        self.mtp_b_i_individual = mtp_b_i_individual
        self.mtp_b_r_individual = mtp_b_r_individual
        self.flat_ai = flat_ai
        self.flat_ar = flat_ar
        self.flat_bi = flat_bi
        self.flat_br = flat_br
        self.number_of_iterations = number_of_iterations
        self.its = None

        
        mtp_ai = self.alpha_i_mp * self.mtp_a_i_individual # Multiplier alpha 
        mtp_ar = self.alpha_r_mp * self.mtp_a_r_individual # Multiplier alpha 
        mtp_bi = self.beta_i_mp * self.mtp_b_i_individual # Multiplier beta 
        mtp_br = self.beta_r_mp * self.mtp_b_r_individual # Multiplier beta 

        self.constants = {
            #Automatic, Social
            # DIVIDED BY 46, as 4.6 is the average in years between examinations, resulting in 10 iterations per year - so to run 5 years number_of_iterations should be 50
            'h_m': np.array([0.2108*mtp_ai[0] + flat_ai, 0.0202 * mtp_bi[0] + flat_bi]) / 46, 
            'h_d': np.array([0.0081*mtp_ai[1] + flat_ai, 0.0000 * mtp_bi[1] + flat_bi]) / 46,
            'm_h': np.array([0.1942*mtp_ar[0] + flat_ar, 0.0348 * mtp_br[0] + flat_br]) / 46,
            'm_d': np.array([0.0839*mtp_ai[2] + flat_ai, 0.0357 * mtp_bi[2] + flat_bi]) / 46,
            'd_h': np.array([0.0558*mtp_ar[1] + flat_ar, 0.0190 * mtp_br[1] + flat_br]) / 46,
            'd_m': np.array([0.2997*mtp_ar[2] + flat_ar, 0.0000 * mtp_br[2] + flat_br]) / 46,  # Social rate from A-> M ~ H
            'h_m_d': np.array([0.0271*mtp_bi[3] + flat_bi]) / 46,
        }

        initial_state = {
            'state': list(nx.get_node_attributes(self.g, 'd_state_ego').values())
        }
        
        self.model.constants = self.constants
        self.model.set_states(['state'])
        self.model.add_update(self.update_state, {'constants': self.model.constants})
        self.model.set_initial_state(initial_state, {'constants': self.model.constants})

        self.correlations = PropertyFunction(
            'correlations',
            self.get_spatial_correlation,
            10,
            # {'nodes': self.model.get_nodes_state(self.model.nodes,'state')}
            {}
        )
        self.model.add_property_function(self.correlations)

    
    def update_state(self, constants):
        state = self.model.get_state('state')
        adjacency = self.model.get_adjacency()

        # Select different states
        healthy_indices = np.where(state == 0)[0]
        mild_indices = np.where(state == 1)[0]
        depressed_indices = np.where(state == 2)[0]

        # Select all neighbours of each state
        healthy_nbs = adjacency[healthy_indices]
        mild_nbs = adjacency[mild_indices]
        depressed_nbs = adjacency[depressed_indices]

        # Get dummy vector for all nodes per state: e.g. n = 6, node 3  and 1 infected: [0,1,0,1,0,0]
        healthy_vec = np.zeros(len(state))
        healthy_vec[healthy_indices] = 1
        mild_vec = np.zeros(len(state))
        mild_vec[mild_indices] = 1
        depressed_vec = np.zeros(len(state))
        depressed_vec[depressed_indices] = 1

        # Get vector of per type ego the adjacency for other type of friends
            # so if there is 3 healthy ego, who each only have 2 mild friends, it will be
            # shape (3, n), filled with zeros except 2 ones when healthy_mild
        healthy_mild = healthy_nbs * mild_vec
        healthy_depressed = healthy_nbs * depressed_vec
        mild_healthy = mild_nbs * healthy_vec
        mild_depressed = mild_nbs * depressed_vec
        depressed_healthy = depressed_nbs * healthy_vec
        depressed_mild = depressed_nbs * mild_vec

        # Get number of friends of certain type for each state
            #  size(n healthy, int)
        num_h_m = healthy_mild.sum(axis = 1)
        num_h_d = healthy_depressed.sum(axis = 1)
        num_m_h = mild_healthy.sum(axis = 1)
        num_m_d = mild_depressed.sum(axis = 1)
        num_d_h = depressed_healthy.sum(axis = 1)
        num_d_m = depressed_mild.sum(axis = 1)

        # Get probability to change state:
            # size n healthy, float
        h_to_m_prob = constants['h_m'][0] + num_h_m * constants['h_m'][1] + num_h_d * constants['h_m_d']
        h_to_d_prob = constants['h_d'][0] + num_h_d * constants['h_d'][1]
        m_to_h_prob = constants['m_h'][0] + num_m_h * constants['m_h'][1]
        m_to_d_prob = constants['m_d'][0] + num_m_d * constants['m_d'][1]
        d_to_h_prob = constants['d_h'][0] + num_d_h * constants['d_h'][1]
        d_to_m_prob = constants['d_m'][0] + num_d_m * constants['d_m'][1]

        # draw uniformly to see who makes transition
        draw_h_m = np.random.random_sample(len(h_to_m_prob))
        draw_h_d = np.random.random_sample(len(h_to_d_prob))
        draw_m_h = np.random.random_sample(len(m_to_h_prob))
        draw_m_d = np.random.random_sample(len(m_to_d_prob))
        draw_d_h = np.random.random_sample(len(d_to_h_prob))
        draw_d_m = np.random.random_sample(len(d_to_m_prob))

        # Node indicators for changing nodes
        nodes_h_to_m = healthy_indices[np.where(h_to_m_prob > draw_h_m)]
        nodes_h_to_d = healthy_indices[np.where(h_to_d_prob > draw_h_d)]
        nodes_m_to_h = mild_indices[np.where(m_to_h_prob > draw_m_h)]
        nodes_m_to_d = mild_indices[np.where(m_to_d_prob > draw_m_d)]
        nodes_d_to_h = depressed_indices[np.where(d_to_h_prob > draw_d_h)]
        nodes_d_to_m = depressed_indices[np.where(d_to_m_prob > draw_d_m)]

        # Update new state variable with changed states
        state[nodes_h_to_m] = 1
        state[nodes_h_to_d] = 2
        state[nodes_m_to_h] = 0
        state[nodes_m_to_d] = 2
        state[nodes_d_to_h] = 0
        state[nodes_d_to_m] = 1

        return {'state': state}

    


    def get_spatial_correlation(self):
        """
        Takes a list of node states (0, 1 or 2) and a networkx graph object and returns 
        the spatial correlation of state 0, 1 and 2.
        """
        state_list = self.model.get_state('state')
        # Get total number of nodes
        num_nodes = len(state_list)
        
        # Calculate the fraction of each state
        state0_frac = np.count_nonzero(state_list == 0) / num_nodes
        state1_frac = np.count_nonzero(state_list == 1) / num_nodes
        state2_frac = np.count_nonzero(state_list == 2) / num_nodes
        
        # Initialize spatial correlation values for each state
        state0_corr = 0
        state1_corr = 0
        state2_corr = 0
        
        # Iterate through all edges and calculate the spatial correlation of each state
        for edge in self.g.edges:
            i, j = edge
            if state_list[i] == 0 and state_list[j] == 0:
                state0_corr += 1
            elif state_list[i] == 1 and state_list[j] == 1:
                state1_corr += 1
            elif state_list[i] == 2 and state_list[j] == 2:
                state2_corr += 1
        
        # Normalize by the total number of edges and return the spatial correlations
        total_edges = self.g.number_of_edges()
        state0_corr /= total_edges
        state1_corr /= total_edges
        state2_corr /= total_edges
        
        return [state0_corr, state1_corr, state2_corr]


    def simulate(self, custom_iterations=None):
        # Check if user provided custom_iterations
        if custom_iterations:
            self.its = self.model.simulate(custom_iterations)
        else:
            self.its = self.model.simulate(self.number_of_iterations)

        # Get all states for all iterations
        iterations = self.its['states'].values()

        H = [np.count_nonzero(it == 0) for it in iterations]
        M = [np.count_nonzero(it == 1) for it in iterations]
        D = [np.count_nonzero(it == 2) for it in iterations]

        # Return number of ppl in each state for last iteration
        return np.array((H, M, D))

# %%

# amha_class = AmhaModel(G)
# %%
# amha_class.simulate(400)
#%%
# cor_dict = amha_class.model.get_properties()
# val = np.array(cor_dict['correlations'])

# # %%
# plt.plot(val[:,0])
# plt.plot(val[:,1])
# plt.plot(val[:,2])
# # %%