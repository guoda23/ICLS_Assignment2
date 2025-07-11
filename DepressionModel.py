#%%
# from init_rcParams import set_mpl_settings
# set_mpl_settings()
#%%
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from dynsimf.models.Model import Model
from dynsimf.models.Model import ModelConfiguration

from dynsimf.models.components.PropertyFunction import PropertyFunction

class DepressionModel:
        #TODO: group therapy effect: increase alpha_r_mp
        #TODO: introduce social buffer effect
        #TODO: add weighted ties
    

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
        adjacency = nx.to_numpy_array(self.g, weight='weight')

        healthy_indices = np.where(state == 0)[0]
        mild_indices    = np.where(state == 1)[0]
        depressed_indices = np.where(state == 2)[0]

        healthy_vec   = (state == 0).astype(float)
        mild_vec      = (state == 1).astype(float)
        depressed_vec = (state == 2).astype(float)
    
        num_h_m = adjacency @ mild_vec        # exposure to Mild
        num_h_d = adjacency @ depressed_vec   # exposure to Depressed
        num_m_h = adjacency @ healthy_vec
        num_m_d = adjacency @ depressed_vec
        num_d_h = adjacency @ healthy_vec
        num_d_m = adjacency @ mild_vec

        # Get probability to change state:
            # size n healthy, float
        h_to_m_prob = constants['h_m'][0] + num_h_m * constants['h_m'][1] + num_h_d * constants['h_m_d']
        h_to_d_prob = constants['h_d'][0] + num_h_d * constants['h_d'][1]
        m_to_h_prob = constants['m_h'][0] + num_m_h * constants['m_h'][1]
        m_to_d_prob = constants['m_d'][0] + num_m_d * constants['m_d'][1]
        d_to_h_prob = constants['d_h'][0] + num_d_h * constants['d_h'][1]
        d_to_m_prob = constants['d_m'][0] + num_d_m * constants['d_m'][1]

        draw_h_m = np.random.random_sample(len(healthy_indices))
        draw_h_d = np.random.random_sample(len(healthy_indices))
        draw_m_h = np.random.random_sample(len(mild_indices))
        draw_m_d = np.random.random_sample(len(mild_indices))
        draw_d_h = np.random.random_sample(len(depressed_indices))
        draw_d_m = np.random.random_sample(len(depressed_indices))

        nodes_h_to_m = healthy_indices[np.where(h_to_m_prob[healthy_indices] > draw_h_m)]
        nodes_h_to_d = healthy_indices[np.where(h_to_d_prob[healthy_indices] > draw_h_d)]
        nodes_m_to_h = mild_indices[np.where(m_to_h_prob[mild_indices] > draw_m_h)]
        nodes_m_to_d = mild_indices[np.where(m_to_d_prob[mild_indices] > draw_m_d)]
        nodes_d_to_h = depressed_indices[np.where(d_to_h_prob[depressed_indices] > draw_d_h)]
        nodes_d_to_m = depressed_indices[np.where(d_to_m_prob[depressed_indices] > draw_d_m)]

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