# several custom pytorch transform callable classes
import more_itertools as miter


# the API of more_itertools should be useful:
# https://more-itertools.readthedocs.io/en/stable/index.html


class SlidingWindow(object):
    def __init__(self, output_size, stride, fillvalue):
        assert output_size >= stride
        self.window_size = output_size
        self.stride = stride
        self.fillvalue = fillvalue

    def __call__(self, sample):
        """
        :param sample: is a dict of feature sequences
        :return:
            if output size is greater than the seq size
        """
        output_dict = {}
        for key in sample.keys():
            seq = sample[key]
            patches = list(miter.windowed(seq, n=self.window_size,
                                          step=self.stride,
                                          fillvalue=self.fillvalue))
            output_dict[key] = patches
        return output_dict


class RandomCrop(object):
    """
    Crop randomly the sequence in a sample
    """

    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def __call__(self, sample):
        pass


class Padding(object):
    def __init__(self, output_size, side=None, fillvalue=None):
        assert isinstance(side, str)
        self.side = side
        self.output_size = output_size
        self.fillvalue = fillvalue

    def __call__(self, sample):
        output_dict = {}
        for key in sample.keys():
            seq = sample[key]
            if self.side in ['right', None]:
                output = list(miter.padded(seq, fillvalue=self.fillvalue, n=self.output_size))
            elif self.side is 'left':
                output = list(miter.padded(seq[::-1], fillvalue=self.fillvalue, n=self.output_size))
                output = output[::-1]
            else:
                raise AttributeError
            output_dict[key] = output
        return output_dict
