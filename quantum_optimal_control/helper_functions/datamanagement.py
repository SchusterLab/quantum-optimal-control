"""
data management library used for the Schuster lab experiments
originally written by: Phil Reinhold & David Schuster
"""

import numpy as np
import h5py
import json

class H5File(h5py.File):
    def __init__(self, *args, **kwargs):
        h5py.File.__init__(self, *args, **kwargs)
        # self.attrs["_script"] = open(sys.argv[0], 'r').read()
        # if self.mode is not 'r':
        # self.attrs["_script"] = get_script()
        # if not read-only or existing then save the script into the .h5
        # Maybe should take this automatic feature out and just do it when you want to
        # Automatic feature taken out. Caused more trouble than convenience. Ge Yang
        # if 'save_script' in kwargs:
        # save_script = kwargs['save_script']
        # else:
        # save_script = True
        # if (self.mode is not 'r') and ("_script" not in self.attrs) and (save_script):
        # self.save_script()
        self.flush()

    # Methods for proxy use    
    def _my_ds_from_path(self, dspath):
        """returns the object (dataset or group) specified by dspath"""
        branch = self
        for ds in dspath:
            branch = branch[ds]
        return branch

    def _my_assign_dset(self, dspath, ds, val):
        print 'assigning', ds, val
        branch = self._my_ds_from_path(dspath)
        branch[ds] = val

    def _get_dset_array(self, dspath):
        """returns a pickle-safe array for the branch specified by dspath"""
        branch = self._my_ds_from_path(dspath)
        if isinstance(branch, h5py.Group):
            return 'group'
        else:
            return (H5Array(branch), dict(branch.attrs))

    def _get_attrs(self, dspath):
        branch = self._my_ds_from_path(dspath)
        return dict(branch.attrs)

    def _set_attr(self, dspath, item, value):
        branch = self._my_ds_from_path(dspath)
        branch.attrs[item] = value

    def _call_with_path(self, dspath, method, args, kwargs):
        branch = self._my_ds_from_path(dspath)
        return getattr(branch, method)(*args, **kwargs)

    def _ping(self):
        return 'OK'

    def set_range(self, dataset, xmin, xmax, ymin=None, ymax=None):
        if ymin is not None and ymax is not None:
            dataset.attrs["_axes"] = ((xmin, xmax), (ymin, ymax))
        else:
            dataset.attrs["_axes"] = (xmin, xmax)

    def set_labels(self, dataset, x_lab, y_lab, z_lab=None):
        if z_lab is not None:
            dataset.attrs["_axes_labels"] = (x_lab, y_lab, z_lab)
        else:
            dataset.attrs["_axes_labels"] = (x_lab, y_lab)

    def append_line(self, dataset, line, axis=0):
        if isinstance(dataset,unicode): dataset=str(dataset)
        if isinstance(dataset, str):
            try:
                dataset = self[dataset]
            except:
                shape, maxshape = (0, len(line)), (None, len(line))
                if axis == 1:
                    shape, maxshape = (shape[1], shape[0]), (maxshape[1], maxshape[0])
                self.create_dataset(dataset, shape=shape, maxshape=maxshape, dtype='float64')
                dataset = self[dataset]
        shape = list(dataset.shape)
        shape[axis] = shape[axis] + 1
        dataset.resize(shape)
        if axis == 0:
            dataset[-1, :] = line
        else:
            dataset[:, -1] = line
        self.flush()

    def append_pt(self, dataset, pt):
        if isinstance(dataset,unicode): dataset=str(dataset)
        if isinstance(dataset, str) :
            try:
                dataset = self[dataset]
            except:
                self.create_dataset(dataset, shape=(0,), maxshape=(None,), dtype='float64')
                dataset = self[dataset]
        shape = list(dataset.shape)
        shape[0] = shape[0] + 1
        dataset.resize(shape)
        dataset[-1] = pt
        self.flush()

    def note(self, note):
        """Add a timestamped note to HDF file, in a dataset called 'notes'"""
        ts = datetime.datetime.now()
        try:
            ds = self['notes']
        except:
            ds = self.create_dataset('notes', (0,), maxshape=(None,), dtype=h5py.new_vlen(str))

        shape = list(ds.shape)
        shape[0] = shape[0] + 1
        ds.resize(shape)
        ds[-1] = str(ts) + ' -- ' + note
        self.flush()

    def get_notes(self, one_string=False, print_notes=False):
        """Returns notes embedded in HDF file if present.
        @param one_string=False if True concatenates them all together
        @param print_notes=False if True prints all the notes to stdout
        """
        try:
            notes = list(self['notes'])
        except:
            notes = []
        if print_notes:
            print '\n'.join(notes)
        if one_string:
            notes = '\n'.join(notes)
        return notes

    def add_data(self, f, key, data):
        data = np.array(data)
        try:
            f.create_dataset(key, shape=data.shape,
                             maxshape=tuple([None] * len(data.shape)),
                             dtype=str(data.dtype))
        except RuntimeError:
            del f[key]
            f.create_dataset(key, shape=data.shape,
                             maxshape=tuple([None] * len(data.shape)),
                             dtype=str(data.dtype))
        f[key][...] = data

    def append_data(self, f, key, data, forceInit=False):
        """
        the main difference between append_pt and append is thta
        append takes care of highier dimensional data, but not append_pt
        """

        data = np.array(data)
        try:
            f.create_dataset(key, shape=tuple([1] + list(data.shape)),
                             maxshape=tuple([None] * (len(data.shape) + 1)),
                             dtype=str(data.dtype))
        except RuntimeError:
            if forceInit == True:
                del f[key]
                f.create_dataset(key, shape=tuple([1] + list(data.shape)),
                                 maxshape=tuple([None] * (len(data.shape) + 1)),
                                 dtype=str(data.dtype))
            dataset = f[key]
            Shape = list(dataset.shape)
            Shape[0] = Shape[0] + 1
            dataset.resize(Shape)

        dataset = f[key]
        try:
            dataset[-1, :] = data
        except TypeError:
            dataset[-1] = data
            # Usage require strictly same dimensionality for all data appended.
            # currently I don't have it setup to return a good exception, but should

    def add(self, key, data):
        self.add_data(self, key, data)

    def append(self, dataset, pt):
        self.append_data(self, dataset, pt)

    # def save_script(self, name="_script"):
    # self.attrs[name] = get_script()
    def save_dict(self, dict, group='/'):
        if group not in self:
            self.create_group(group)
        for k in dict.keys():
            self[group].attrs[k] = dict[k]

    def get_dict(self, group='/'):
        d = {}
        for k in self[group].attrs.keys():
            d[k] = self[group].attrs[k]
        return d

    get_attrs = get_dict
    save_attrs = save_dict


    def save_settings(self, dic, group='settings'):
        self.save_dict(dic, group)

    def load_settings(self, group='settings'):
        return self.get_dict(group)

    def load_config(self):
        if 'config' in self.attrs.keys():
            return AttrDict(json.loads(self.attrs['config']))
        else:
            return None