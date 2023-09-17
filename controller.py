import do_mpc
import numpy as np
import casadi as ca
from time import sleep
from itertools import chain
from typing import Dict, List, Optional


from core import AbstractController
from llm import Objective, Optimization
from config.config import BaseNMPCConfig

class BaseController(AbstractController):

  def __init__(self, robot_names:List[str], cfg=BaseNMPCConfig) -> None:
    super().__init__(cfg)

    # init names of robots
    self.robot_names = robot_names

    # init model dynamics
    self.init_model()

    # init controller
    self.init_controller()

    # init variables and expressions
    self.init_expressions()

    # gripper fingers offset for constraints 
    self.gripper_offsets = [np.array([0., -0.048, 0.]), np.array([0., 0.048, 0.]), np.array([0., 0., 0.048])]


  def init_model(self):
    # inti do_mpc model
    self.model = do_mpc.model.Model(self.cfg.model_type) 

    # simulation time
    self.t = self.model.set_variable('parameter', 't')

    self.xi= []     # home position
    self.x = []     # gripper position (x,y,z)
    self.psi = []   # gripper psi (rotation around z axis)
    self.dx = []    # gripper velocity (vx, vy, vz)
    self.dpsi = []  # gripper rotational speed
    self.u = []     # gripper control (=velocity)
    self.u_psi = [] # gripper rotation control (=rotational velocity)

    for i, r in enumerate(self.robot_names):
      # position (x, y, z)
      self.x.append(self.model.set_variable(var_type='_x', var_name=f'x{r}', shape=(self.cfg.nx,1)))
      self.psi.append(self.model.set_variable(var_type='_x', var_name=f'psi{r}', shape=(1,1)))
      self.dx.append(self.model.set_variable(var_type='_x', var_name=f'dx{r}', shape=(self.cfg.nx,1)))
      self.dpsi.append(self.model.set_variable(var_type='_x', var_name=f'dpsi{r}', shape=(1,1)))
      self.u.append(self.model.set_variable(var_type='_u', var_name=f'u{r}', shape=(self.cfg.nu,1)))
      self.u_psi.append(self.model.set_variable(var_type='_u', var_name=f'u_psi{r}', shape=(1,1)))
      # system dynamics
      self.model.set_rhs(f'x{r}', self.x[i] + self.dx[i] * self.cfg.dt)
      self.model.set_rhs(f'psi{r}', self.psi[i] + self.dpsi[i] * self.cfg.dt)
      self.model.set_rhs(f'dx{r}', self.u[i])
      self.model.set_rhs(f'dpsi{r}', self.u_psi[i])
    
    # setup model
    self.model.setup()
  
  def set_objective(self, mterm: ca.SX = ca.DM([[0]])): # TODO: not sure if ca.SX is the right one
    # objective terms
    mterm = mterm # TODO: add psi reference like this -> 0.1*ca.norm_2(-1-ca.cos(self.psi_right))**2
    lterm = 0.4*mterm
    # state objective
    self.mpc.set_objective(mterm=mterm, lterm=lterm)
    # input objective
    u_kwargs = {f'u{r}':1. for r in self.robot_names} | {f'u_psi{r}':1. for r in self.robot_names} 
    self.mpc.set_rterm(**u_kwargs)

  def set_constraints(self, nlp_constraints: Optional[List[ca.SX]] = None):

    for r in self.robot_names:
      # base constraints (state)
      self.mpc.bounds['lower','_x', f'x{r}'] = np.array([-3., -3., 0.0]) # stay above table
      #self.mpc.bounds['upper','_x', f'psi{r}'] = np.pi/2 * np.ones((1, 1))   # rotation upper bound
      #self.mpc.bounds['lower','_x', f'psi{r}'] = -np.pi/2 * np.ones((1, 1))  # rotation lower bound

      # base constraints (input)
      self.mpc.bounds['upper','_u', f'u{r}'] = self.cfg.hu * np.ones((self.cfg.nu, 1))  # input upper bound
      self.mpc.bounds['lower','_u', f'u{r}'] = self.cfg.lu * np.ones((self.cfg.nu, 1))  # input lower bound
      self.mpc.bounds['upper','_u', f'u_psi{r}'] = np.pi * np.ones((1, 1))   # input upper bound
      self.mpc.bounds['lower','_u', f'u_psi{r}'] = -np.pi * np.ones((1, 1))  # input lower bound

    if nlp_constraints == None: 
      return

    for i, constraint in enumerate(nlp_constraints):
      self.mpc.set_nl_cons(f'const{i}', expr=constraint, ub=0., 
                          soft_constraint=True, 
                          penalty_term_cons=self.cfg.penalty_term_cons)

  def init_mpc(self):
    # init mpc model
    self.mpc = do_mpc.controller.MPC(self.model)
    # setup params
    setup_mpc = {'n_horizon': self.cfg.T, 't_step': self.cfg.dt, 'store_full_solution': False}
    # setup mpc
    self.mpc.set_param(**setup_mpc)
    self.mpc.settings.supress_ipopt_output() # => verbose = False

  def init_controller(self):
    # init
    self.init_mpc()
    # set functions
    self.set_objective()
    self.set_constraints()
    # setup
    self.mpc.set_uncertainty_values(t=np.array([0.])) # init time to 0
    self.mpc.setup()
    self.mpc.set_initial_guess()

  def init_expressions(self):
    # init variables for python evaluation
    self.eval_variables = {"ca":ca, "np":np, "t":self.t} # python packages

    self.R = [] # rotation matrix for angle around z axis
    for i in range(len(self.robot_names)):
      # rotation matrix
      self.R.append(np.array([[ca.cos(self.psi[i]), -ca.sin(self.psi[i]), 0],
                              [ca.sin(self.psi[i]), ca.cos(self.psi[i]), 0],
                              [0, 0, 1.]]))

  def set_t(self, t:float):
    """ Update the simulation time of the MPC controller"""
    self.mpc.set_uncertainty_values(t=np.array([t]))

  def set_x0(self, observation: Dict[str, np.ndarray]):
    x0 = []
    for r in self.robot_names: # TODO set names instead of robot_0 in panda
      obs = observation[f'robot{r}'] # observation of each robot
      x = obs[:3]
      psi = np.array([obs[5]])
      dx = obs[6:9]
      x0.append(np.concatenate((x, psi, dx, [0]))) # TODO dpsi is harcoded to 0 here
    # set x0 in MPC
    self.mpc.x0 = np.concatenate(x0)

  def reset(self, observation: Dict[str, np.ndarray]) -> None:
    """
      observation: robot observation from simulation containing position, angle and velocities 
    """
    # TODO
    self.set_x0(observation)
    return

  def _eval(self, code_str: str, observation: Dict[str, np.ndarray], offset=np.zeros(3)):
    #TODO the offset is still harcoded
    # put together variables for python code evaluation:    
    robots_states = {}
    for i, r in enumerate(self.robot_names):
      R = np.array([[ca.cos(self.psi[i]), -ca.sin(self.psi[i]), 0],
                    [ca.sin(self.psi[i]), ca.cos(self.psi[i]), 0],
                    [0, 0, 1.]])
      robots_states[f'x{r}'] = self.x[i] + self.R[i]@offset
      robots_states[f'dx{r}'] = self.dx[i]
    
    eval_variables = self.eval_variables | robots_states | observation
    # evaluate code
    evaluated_code = eval(code_str, eval_variables)
    return evaluated_code

  def _solve(self) -> List[np.ndarray]:
    """ Returns a list of conntrols, 1 for each robot """
    # solve mpc at state x0
    u0 = self.mpc.make_step(self.mpc.x0).squeeze()
    # compute action for each robot
    action = []
    for i in range(len(self.robot_names)):
      ee_displacement = u0[4*i:4*i+3]     # positon control
      psi_rotation = [u0[4*i+3]]            # rotation control
      theta_gamma_rotation = np.zeros(2)  # TODO no roll and pith control now
      action.append(np.concatenate((ee_displacement, theta_gamma_rotation, psi_rotation)))
    
    return action

  def step(self):
    if not self.mpc.flags['setup']:
      return [np.zeros(6) for i in range(len(self.robot_names))]  # TODO 6 is hard-coded here
    return self._solve()


class ObjectiveNMPC(BaseController):

  def apply_gpt_message(self, objective: Objective, observation: Dict[str, np.ndarray]) -> None:
    # init mpc newly
    self.init_mpc()
    # apply constraint function
    self.set_objective(self._eval(objective.objective, observation))
    # set base constraint functions
    self.set_constraints()
    # setup
    self.mpc.setup()
    self.mpc.set_initial_guess()
    return 

class OptimizationNMPC(BaseController):

  def apply_gpt_message(self, optimization: Optimization, observation: Dict[str, np.ndarray]) -> None:
    # init mpc newly
    self.init_mpc()
    # apply constraint function
    # NOTE use 1e-6 when doing task L 
    regulatization = 0#1 * ca.norm_2(self.dpsi)**2 #+ 0.1 * ca.norm_2(self.psi - np.pi/2)**2
    self.set_objective(self._eval(optimization.objective, observation) + regulatization)
    # set base constraint functions
    constraints = [[*map(lambda const: self._eval(c, observation, const), self.gripper_offsets)] for c in optimization.constraints]
    self.set_constraints(list(chain(*constraints)))
    # setup
    self.mpc.set_uncertainty_values(t=np.array([0.])) # TODO this is badly harcoded
    self.mpc.setup()
    self.mpc.set_initial_guess()
    return 

ControllerOptions = {
  "objective": ObjectiveNMPC,
  "optimization": OptimizationNMPC
}


