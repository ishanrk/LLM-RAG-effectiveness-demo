# sourced from https://towardsdatascience.com/jane-the-discoverer-enhancing-causal-discovery-with-large-language-models-causal-python-564a63425c93
# this demo elaborates the effectiveness of LLM+RAG in causal discovery as opposed to sole LLM
from itertools import combinations
import os
os.environ['OPENAI_API_KEY'] = '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'

import numpy as np
from scipy import linalg 
from scipy import stats 

import matplotlib.pyplot as plt

from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType

from langchain.chat_models import ChatOpenAI

from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.algorithms import PC

from castle.common.priori_knowledge import PrioriKnowledge
os.environ

all_vars = {
    'drought_risk': 0,
    'salinity': 1,
    'temperature': 2,
    'crop_yield': 3,
    'alkaline_index': 4
}

SAMPLE_SIZE = 1000

drought_risk = stats.halfnorm.rvs(scale=2000, size=SAMPLE_SIZE)
temperature = 25 - drought_risk / 100 + stats.norm.rvs(
    loc=0,
    scale=2,
    size=SAMPLE_SIZE
)

alkaline_index = stats.halfnorm.rvs(size=SAMPLE_SIZE)

salinity = np.clip(
    1 - drought_risk / 8000 
    - temperature / 50 
    + stats.norm.rvs(size=SAMPLE_SIZE) / 20,
    0, 
    1)

crop_yield = np.clip(
    drought_risk / 20000 
    + np.abs(temperature) / 100 
    - salinity / 5 
    - alkaline_index / 5
    + stats.norm.rvs(size=SAMPLE_SIZE) / 10,
    0,
    1
)

dataset = np.stack(
    [
        drought_risk,
        salinity,
        temperature,
        crop_yield,
        alkaline_index
    ]
).T

true_dag = np.array(
    [
        [0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0]
    ]
)

pc = PC(variant='stable')
pc.learn(dataset)

# Vizualize
GraphDAG(
    est_dag=pc.causal_matrix, 
    true_dag=true_dag)

plt.show()

# Compute metrics
metrics = MetricsDAG(
    B_est=pc.causal_matrix, 
    B_true=true_dag)

print(metrics.metrics)

priori_knowledge = PrioriKnowledge(n_nodes=len(all_vars))

llm = ChatOpenAI(
    temperature=0, 
    model='gpt-4')

tools = load_tools(
    [
        "wikipedia"
    ], 
    llm=llm)

agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=False)

def get_llm_info(llm, agent, var_1, var_2):
    
    out = agent(f"Does {var_1} cause {var_2} or the other way around?\
    We assume the following definition of causation:\
    if we change A, B will also change.\
    The relationship does not have to be linear or monotonic.\
    We are interested in all types of causal relationships, including\
    partial and indirect relationships, given that our definition holds.\
    ")
    
    print(out)
    
    pred = llm.predict(f'We assume the following definition of causation:\
    if we change A, B will also change.\
    Based on the following information: {out["output"]},\
    print (0,1) if {var_1} causes {var_2},\
    print (1, 0) if {var_2} causes {var_1}, print (0,0)\
    if there is no causal relationship between {var_1} and {var_2}.\
    Finally, print (-1, -1) if you don\'t know. Importantly, don\'t try to\
    make up an answer if you don\'t know.')
    
    print(pred)

    
    return pred

print('\nRunning PC')
# Instantiate the model with expert knowledge
pc_priori = PC(
    priori_knowledge=priori_knowledge,
    variant='stable'
)

# Learn
pc_priori.learn(dataset)

GraphDAG(
    est_dag=pc_priori.causal_matrix, 
    true_dag=true_dag)

plt.show()

# Compute metrics
metrics = MetricsDAG(
    B_est=pc_priori.causal_matrix, 
    B_true=true_dag)

print(metrics.metrics)

for var_1, var_2 in combinations(all_vars.keys(), r=2):
    print(var_1, var_2)
    out = get_llm_info(llm, agent, var_1, var_2)
    if out=='(0,1)':
        priori_knowledge.add_required_edges(
            [(all_vars[var_1], all_vars[var_2])]
        )
        
        priori_knowledge.add_forbidden_edges(
            [(all_vars[var_2], all_vars[var_1])]
        )

    elif out=='(1,0)':
        priori_knowledge.add_required_edges(
            [(all_vars[var_2], all_vars[var_1])]
        )
        priori_knowledge.add_forbidden_edges(
            [(all_vars[var_1], all_vars[var_2])]
        )


print('\nLLM knowledge vs true DAG')
priori_dag = np.clip(priori_knowledge.matrix, 0, 1)

GraphDAG(
    est_dag=priori_dag, 
    true_dag=true_dag)

plt.show()
