from lux.kit import obs_to_game_state, GameState
from lux.config import EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
import numpy as np
import sys

class Agent():
    def __init__(self,player:str,env_cfg:EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env.cfg:EnvConfig = env_cfg

        def early_setup(self, step:int, obs, remainingOverageTime:int = 60):
            if step == 0:
                # bid 0 to not waste resources bidding and declare as the default faction
                return dict(faction="AlphaStrike", bid=0)
            else:
                game_state = obs_to_game_state(step, self.env_cfg, obs)
                #factory placement period

                water_left = game_state.teams[self.player].water
                metal_left = game_state.teams[self.player].metal
                factories_to_place = game_state.teams[self.player].factories_to_place
                my_turn_to_place = my_turn_to_place_factory(game_state.teams[self.player].place_first, step)
                if factories_to_place > 0 and my_turn_to_place:
                    potential_spawn = np.array(list(zip(*np.where(obs["board"]["valid_spawn_mask"] == 1))))
                    ice_map = game_state.board.ice
                    ice_tile_locations = np.argwhere(ice_map == 1)
                    spawnpoint_closest_to_ice = potential_spawn[0]
                    spawnpoint_ice_closest_distance = 10000000

                    for spawnpoint in potential_spawn:
                        spawnpoint_ice_distances = np.mean((spawnpoint - ice_tile_locations)**2,1)
                        spawnpoint_ice_distance = np.min(spawnpoint_ice_distances)
                        if spawnpoint_ice_distance < spawnpoint_ice_closest_distance:
                            spawnpoint_closest_to_ice = spawnpoint
                    
                    return dict(spawn = spawnpoint_closest_to_ice, metal = metal_left/factories_to_place, water = water_left/factories_to_place)
                return dict()