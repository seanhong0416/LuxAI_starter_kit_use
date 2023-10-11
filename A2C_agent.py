from lux.kit import obs_to_game_state, GameState
from lux.config import EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
from A2C_model import A2C_Net
import numpy as np
import sys

class A2C_Agent():
    def __init__(self,player:str,env_cfg:EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.a2c_net = A2C_Net()
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
    
    def act(self, step:int, obs, remainingOverageTime:int = 60):
        actions = dict()
        game_state = obs_to_game_state(step, self.env_cfg, obs)
        
        #critic addtional input
        total_power = 0
        total_water = 0
        total_unit = 0
        total_ally_lichen = 0

        #actor additional input
        """
        1.unit_type
        2.power
        3.position
        4.cargo.ice
        5.cargo.ore
        total:6 additional input
        """
        actor_additional_input = {}
        
        #factory actions
        ally_factories = game_state.factories[self.player]
        ally_factory_tiles = []
        for unit_id, factory in ally_factories.items():
            if factory.power >= self.env_cfg.ROBOTS["LIGHT"].POWER_COST and\
            factory.cargo.metal >= self.env_cfg.ROBOTS["LIGHT"].METAL_COST:
                actions[unit_id] = factory.build_light
            if factory.water_cost(game_state) <= factory.cargo.water/5 -200:
                actions[unit_id] = factory.water()
            ally_factory_tiles += factory.pos
            #get value of total power and total water
            total_power += factory.power
            total_water += factory.water
        
        #building input for cnn 
        #get enemy factory placements
        enemy_factories = game_state.factories[self.opp_player]
        enemy_factory_tiles = []
        for unit_id, factory in enemy_factories.items():
            enemy_factory_tiles += factory.pos
        #building factory map, ally factory 1, enemy factory = -1
        factory_map = np.zeros((64,64))
        for pos in ally_factory_tiles:
            factory[tuple(pos)] = 1
        for pos in enemy_factory_tiles:
            factory[tuple(pos)] = -1

        #building unit maps
        unit_map = np.zeros((64,64))
        ally_units = game_state.units[self.player]
        for unit_id, unit in ally_units.items():
            if unit.unit_type == "HEAVY":
                unit_map[tuple(unit.pos)] = 10
            else:
                unit_map[tuple(unit.pos)] = 1
            actor_additional_input[unit_id] = np.array((unit.pos[0],
                                                        unit.pos[1],
                                                        unit.unit_type,
                                                        unit.power,
                                                        unit.cargo.ice,
                                                        unit.cargo.ice))
        #get value of total_unit
        total_unit = np.sum(unit_map)
        enemy_units = game_state.units[self.opp_player]
        for unit_id, unit in enemy_units.items():
            if unit.unit_type == "HEAVY":
                unit_map[tuple(unit.pos)] = -10
            else:
                unit_map[tuple(unit.pos)] = -1

        #get total_lichen
        factory_strains = game_state.teams[self.player].factory_strains
        lichen_strain_map = game_state.board.lichen_strains
        for row in lichen_strain_map:
            for tile in row:
                if tile in factory_strains:
                    total_ally_lichen += 1

        critic_additional_input = np.array([total_power,
                                            total_water,
                                            total_unit,
                                            total_ally_lichen])

        maps = np.concatenate((game_state.board.ice,
                               game_state.board.ore,
                               game_state.board.rubble,
                               game_state.board.lichen,
                               factory_map,
                               unit_map
                               ))
        
        net_output_action, net_output_value = self.a2c_net(maps, actor_additional_input, critic_additional_input)
        for unit_id, action_code in net_output_action.items():
            actions[unit_id] = [self.translate_action(ally_units[unit_id],action_code)]
        
        return actions

    def translate_action(self, unit, action_code):
        recharge_amount = 0
        if unit.unit_type == "LIGHT":
            recharge_amount = 1
        else:
            recharge_amount = 10

        action_dictionary = {0:unit.move(0),
                             1:unit.move(1),
                             2:unit.move(2),
                             3:unit.move(3),
                             4:unit.move(4),
                             5:unit.transfer(1),
                             6:unit.transfer(2),
                             7:unit.transfer(3),
                             8:unit.transfer(4),
                             9:unit.dig(),
                             10:unit.pickup(),
                             11:unit.self_destruct(),
                             12:unit.recharge(recharge_amount)}
        
        return action_dictionary[action_code]
    