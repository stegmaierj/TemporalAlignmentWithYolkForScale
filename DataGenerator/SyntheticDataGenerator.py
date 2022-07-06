import numpy as np

class SyntheticDataGenerator():
    '''
    Generator for synthetic data based on real data. Possibilities to scale, shift, rotate, scatter or choose random points
    '''
    def createSyntheticSampleWithTranslateAndScale(self, randomgenerator, pc, middle_point, translate_def, scale_def):
        '''
        Creates a data sample that is randomly scaled and shifted, the middle point is used to move the data sample to have a (0,0,0) middlepoint for correct scaling
        randomgenerator: a np.random.Generator
        pc: a dict with minimum x,y,z as keys other fielda are preserved
        middle_point: the middlepoint of the point cloud
        translate_def: [mean, std] of translation in mycro meters
        scale_def: [mean, std] of scaling (1 would mean no scaling at all, values in (0,1) make it smaller, [1,inf) make it bigger)

        returns new point cloud with changed values
        '''
        shifted_pc = pc.copy()
        #move according to middle point:
        shifted_pc['x'] = shifted_pc['x'] - middle_point[0]
        shifted_pc['y'] = shifted_pc['y'] - middle_point[1]
        shifted_pc['z'] = shifted_pc['z'] - middle_point[2]

        shifted_pc, scale_factor = self.scaleRandomly(randomgenerator, shifted_pc, scale_def[0], scale_def[1], True)
        shifted_pc, translate_factor = self.shiftRandomly(randomgenerator, shifted_pc, translate_def[0], translate_def[1], True)
        return (shifted_pc, scale_factor, translate_factor)

#-------------------------------------------------------------------------------


    def scaleRandomly(self, randomgenerator, pc, scale_mean, scale_derivation, inPlace=False):
        scale_factor = randomgenerator.normal(scale_mean, scale_derivation)
        if inPlace:
            shifted_pc = pc
        else:
            shifted_pc = pc.copy()
        shifted_pc['x'] = shifted_pc['x']*scale_factor
        shifted_pc['y'] = shifted_pc['y']*scale_factor
        shifted_pc['z'] = shifted_pc['z']*scale_factor
        return (shifted_pc, scale_factor)


#-------------------------------------------------------------------------------

    def rotateRandomly(self, randomgenerator, pc, rotate_mean, rotate_derivation, inPlace=False):
        thetaX = randomgenerator.normal(rotate_mean, rotate_derivation)
        thetaY = randomgenerator.normal(rotate_mean, rotate_derivation)
        thetaZ = randomgenerator.normal(rotate_mean, rotate_derivation)

        if inPlace:
            shifted_pc = pc
        else:
            shifted_pc = pc.copy()
        Rx = np.array([[1, 0,            0],
            [0, np.cos(thetaX), -np.sin(thetaX)],
            [0, np.sin(thetaX),  np.cos(thetaX)]])

        Ry = np.array([[np.cos(thetaY), 0,np.sin(thetaY)],
            [0, 1, 0],
            [-np.sin(thetaY), 0,  np.cos(thetaY)]])

        Rz = np.array([[np.cos(thetaZ), -np.sin(thetaZ),0],
            [np.sin(thetaZ), np.cos(thetaZ), 0],
            [0, 0,  1]])
        R = np.matmul(Rx, np.matmul(Ry, Rz))
        xyz_after_rotatation = R.dot(np.array([shifted_pc['x'], shifted_pc['y'], shifted_pc['z']]))
        shifted_pc['x'] = xyz_after_rotatation[0,:]
        shifted_pc['y'] = xyz_after_rotatation[1,:]
        shifted_pc['z'] = xyz_after_rotatation[2,:]
        return (shifted_pc, [thetaX, thetaY, thetaZ])

#-------------------------------------------------------------------------------

    def shiftRandomly(self,randomgenerator: np.random.Generator, pc, shift_mean, shift_derivation, inPlace=False ):
        translate_x = randomgenerator.normal(shift_mean, shift_derivation)
        translate_y = randomgenerator.normal(shift_mean, shift_derivation)
        translate_z = randomgenerator.normal(shift_mean, shift_derivation)
        if inPlace:
            shifted_pc = pc
        else:
            shifted_pc = pc.copy()
        shifted_pc['x'] = shifted_pc['x'] + translate_x
        shifted_pc['y'] = shifted_pc['y'] + translate_y
        shifted_pc['z'] = shifted_pc['z'] + translate_z
        return (shifted_pc, [translate_x, translate_y, translate_z])


#-------------------------------------------------------------------------------

    def scatterPoints(self, randomgenerator: np.random.Generator, pc, noise_derivation, inPlace = False):
        if inPlace:
            shifted_pc = pc
        else:
            shifted_pc = pc.copy()
        shifted_pc['x'] = shifted_pc['x']+randomgenerator.normal(size=shifted_pc['x'].shape,  scale=noise_derivation )
        shifted_pc['y'] = shifted_pc['y']+randomgenerator.normal(size=shifted_pc['y'].shape,  scale=noise_derivation )
        shifted_pc['z'] = shifted_pc['z']+randomgenerator.normal(size=shifted_pc['z'].shape,  scale=noise_derivation )
        return shifted_pc
        
#-------------------------------------------------------------------------------

    def deletePoints(self, randomgenerator: np.random.Generator, pc, prop_to_be_deleted, inPlace = False):
        '''
        This will lead to inconsistant tracking. As the tracking is not used for registration this is not a problem here. 
        '''
        mask_array_to_stay_in_point_cloud = np.ones(pc['x'].size, dtype=bool)
        inidices_to_be_deleted = np.where((randomgenerator.binomial(size=pc['x'].size, n=1, p= 0.2)==1), True, False)
        mask_array_to_stay_in_point_cloud[inidices_to_be_deleted] = False
        if inPlace:
            shifted_pc = pc
        else:
            shifted_pc = pc.copy()
        for key in pc.keys():
            shifted_pc[key] = shifted_pc[key][mask_array_to_stay_in_point_cloud]

        return shifted_pc
