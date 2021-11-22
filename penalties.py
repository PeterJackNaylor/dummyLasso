import jax.numpy as np

def none(theta):
    return 0

def lasso(theta, lamb=0.5):
    return (lamb * np.absolute(theta)).sum()

def scad(theta, a=0.5):
    pass

def mcp(theta, b):
    pass 

penalty_dic = {
    "None" : none,
    "lasso" : lasso,
    "scad" : scad,
    "MCP": mcp

}