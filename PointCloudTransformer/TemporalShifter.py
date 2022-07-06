import numpy as np

class TemporalShifter():
    '''
    Applies temporal shift map to 4d point cloud with tracking
    '''

    def shift(self, point_cloud, shift_map):
        '''
        Applies the shift. 
        point_cloud: dict with fields x,y,z, time, id, mother_id
        shift_map: list of pairs of frames, that represent a mapping of old to new time frame

        '''
        #initialize fields for new synthetic point cloud:
        shifted_time_array = np.full(point_cloud['time'].shape, -1)
        shifted_mother_id_tmp = np.full(point_cloud['mother_id'].shape, -1)
        new_ids = np.array([])
        new_x = np.array([])
        new_y = np.array([])
        new_z = np.array([])
        new_mother_ids = np.array([])
        new_time = np.array([])
        ids = point_cloud['id']
        current_original_time_point = 0
        current_paired_time_point = 0
        max_id = np.max(point_cloud['id'])
        # apply each time shift in the time shift map:
        for pair in shift_map:
            next_paired_timepoint = pair[1]
            indices_to_shift = point_cloud['time']==pair[0]
            shifted_time_array[indices_to_shift]= pair[1]

            #mother ids should be in the last time frame:
            time_of_mother = pair[0]-1
            indices_of_mother_ids = point_cloud['mother_id'][indices_to_shift]
            #mother ids of elements to be shifted:
            mother_ids = point_cloud['mother_id'][indices_to_shift]

            #in case time frames are deleted, follow the mother ids up until we find one that will actually be in the new point cloud:
            while current_original_time_point < time_of_mother:
                idsorted = np.argsort(ids)
                motheridpos = np.searchsorted(ids[idsorted], mother_ids)
                indices_of_mother_ids = idsorted[motheridpos]
                mother_ids = point_cloud['mother_id'][indices_of_mother_ids]
                time_of_mother = time_of_mother-1
            shifted_mother_id_tmp[indices_to_shift] = mother_ids

            # in case new time frames need to be introduced the tracking information is used to invent intermediate 3d point clouds:
            if current_paired_time_point + 1 < next_paired_timepoint:
                difference_plus_1 = next_paired_timepoint - current_paired_time_point
                number_of_points_to_be_added = np.sum(indices_to_shift)
                #calculate directions:
                mother_ids = point_cloud['mother_id'][indices_to_shift]
                idsorted = np.argsort(ids)
                motheridpos = np.searchsorted(ids[idsorted], mother_ids)
                indices_of_mother_ids = idsorted[motheridpos]

                #direction of movement of points:
                x_directions = point_cloud['x'][indices_of_mother_ids]-point_cloud['x'][indices_to_shift]
                y_directions = point_cloud['y'][indices_of_mother_ids]-point_cloud['y'][indices_to_shift]
                z_directions = point_cloud['z'][indices_of_mother_ids]-point_cloud['z'][indices_to_shift]
                for t in np.arange(difference_plus_1-1):
                    new_ids_to_be_added = np.arange(max_id+1, max_id+number_of_points_to_be_added+1)
                    if t==0:
                        shifted_mother_id_tmp[indices_to_shift]=new_ids_to_be_added
                    new_ids = np.concatenate([new_ids, new_ids_to_be_added])
                    new_x = np.concatenate([new_x,point_cloud['x'][indices_to_shift] + (t+1)*1/difference_plus_1*x_directions])
                    new_y = np.concatenate([new_y,point_cloud['y'][indices_to_shift] + (t+1)*1/difference_plus_1*y_directions])
                    new_z = np.concatenate([new_z,point_cloud['z'][indices_to_shift] + (t+1)*1/difference_plus_1*z_directions])
                    if t == (difference_plus_1-2):
                        new_mother_ids = np.concatenate([new_mother_ids,(point_cloud['id'][indices_of_mother_ids])])
                    else:
                        new_mother_ids = np.concatenate([new_mother_ids,np.arange(max_id+1+number_of_points_to_be_added, max_id+2*number_of_points_to_be_added+1)])
                    max_id = max_id+number_of_points_to_be_added
                    new_time = np.concatenate([new_time, np.full(number_of_points_to_be_added, next_paired_timepoint-t-1)])

            current_original_time_point = pair[0]
            current_paired_time_point = pair[1]

        #cut away non-mapped timeframes:
        indices_not_to_be_cut_away = shifted_time_array != -1
        shifted_x = np.concatenate([point_cloud['x'][indices_not_to_be_cut_away],new_x])
        shifted_y = np.concatenate([point_cloud['y'][indices_not_to_be_cut_away], new_y])
        shifted_z = np.concatenate([point_cloud['z'][indices_not_to_be_cut_away], new_z])
        shifted_id = np.concatenate([point_cloud['id'][indices_not_to_be_cut_away], new_ids])
        shifted_mother_id = np.concatenate([shifted_mother_id_tmp[indices_not_to_be_cut_away], new_mother_ids])
        shifted_time_array_final = np.concatenate([shifted_time_array[indices_not_to_be_cut_away], new_time])
        return {'time':shifted_time_array_final, 'x': shifted_x, 'y': shifted_y, 'z': shifted_z, 'mother_id': shifted_mother_id, 'id': shifted_id }