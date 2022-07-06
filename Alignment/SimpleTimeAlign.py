import numpy as np

class SimpleTimeAlign():

    def align(self, list1, list2, norm):
        '''
        Aligns two monotone increasing lists with each other
        '''

        pairs = np.full(shape=list2.shape[0]+1, fill_value = -1, dtype=int)
        i = 0
        j = 0
        while ((i<(list2.shape[0])) | (j<(list1.shape[0]))):
            
            a = list2[i]
            b = list1[j]
            if (isinstance(a, type(None)) or isinstance(b, type(None))):
                break
            print('{}, {}'.format(i,j))
            if norm(a)<norm(b):
                while (norm(a)<norm(b)) & (i<(list2.shape[0])) :
                    i += 1
                    if i>= (list2.shape[0]-1):
                        break
                    a = list2[i]
                if i>= (list2.shape[0]-1):
                        break
                new_i = np.array([i-1,i])[np.argmin([norm(list2[i-1]-list1[j]), norm(list2[i]-list1[j])])]
                pairs[j] = new_i
                i = new_i+1
                j = j+1
                a = list2[i]
                b = list1[j]
            if norm(b)<norm(a):
                while (norm(b)<norm(a)) & ((j<list1.shape[0])):
                    j += 1
                    if (j>= list1.shape[0]-1):
                        break
                    b = list1[j]
                if j>= (list1.shape[0]-1):
                        break
                new_j = np.array([j-1,j])[np.argmin([norm(list1[j-1]-list2[i]), norm(list1[j]-list2[i])])]
                pairs[new_j] = i
                j = new_j+1
                i = i+1
                a = list2[i]
                b = list1[j]
            if norm(b)==norm(a):
                pairs[j] = i
                j = j+1
                i = i+1
        return np.stack([np.where(pairs>-1)[0], pairs[pairs>-1]], axis=1)