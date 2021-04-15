from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

exp1 = Genotype(
	normal=[('sep_conv_3x3', 1),
		('sep_conv_5x5', 0),
		('sep_conv_3x3', 2),
		('sep_conv_5x5', 1),
		('sep_conv_3x3', 1),
		('sep_conv_3x3', 2),
		('skip_connect', 3),
		('dil_conv_3x3', 0)],
	normal_concat=range(2,6),
	reduce=[('max_pool_3x3', 1),
		('sep_conv_3x3', 0),
		('max_pool_3x3', 2),
		('sep_conv_5x5', 1),
		('sep_conv_5x5', 1),
		('max_pool_3x3', 3),
		('sep_conv_5x5', 1),
		('sep_conv_5x5', 3)],
	reduce_concat=range(2,6)		 
)