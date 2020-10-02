def valid_state(state, car_points, obs_list):
    for obs in obs_list:
        if np.any(obs.points_in_obstacle(car_points.get_points_world_frame(*state))):
            return False
    return True