import numpy
from sscPimega import pi540D, pi135D

########### 540D ###########################

img = numpy.load('SiriusImage3072x3072.npy') * 2**16

distance = 100 #mm

params = {'geo': 'nonplanar', 'opt': False, 'mode': 'virtual', 'fill': False, 'susp': 3}

project = pi540D.dictionary540D(distance, params )

geometry = pi540D.geometry540D( project )

forw = pi540D.forward540D( img , geometry )

rest = pi540D.backward540D( forw , geometry )

forw.astype(numpy.uint32).tofile('rawSimulated540D.b')

# This is the only thing really important for the backend 
geometry['LUT'][0].tofile('y540D.b')
geometry['LUT'][1].tofile('x540D.b')

########### 135D ###########################

img = numpy.load('SiriusImage1536x1536.npy') * 2**16

distance = 100 #mm

params = {'geo': 'nonplanar', 'opt': False, 'mode': 'virtual', 'fill': False, 'susp': 3}

project = pi135D.dictionary135D(distance, params )

geometry = pi135D.geometry135D( project )

forw = pi135D.forward135D( img , geometry )

rest = pi135D.backward135D( forw , geometry )

forw.astype(numpy.uint32).tofile('rawSimulated135D.b')

# This is the only thing really important for the backend 
geometry['LUT'][0].tofile('y135D.b')
geometry['LUT'][1].tofile('x135D.b')
