import json
from typing import Dict
import sys
from argparse import Namespace

from agent import Agent
from lux.config import EnvConfig
from lux.kit import GameState, process_obs, to_json, from_json, process_action, obs_to_game_state