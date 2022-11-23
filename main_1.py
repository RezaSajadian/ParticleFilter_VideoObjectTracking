
#from ParticleFilter_Object_with_mouse_click import ParticleFilter
from ParticleFilter_Module import ParticleFilter

video_file = "/Users/rezasajadian/storage/Training/deep_learning/computer-vision/Courses/Object_Tracking_Using_Particle_Filters/walking.mp4"

config_file = "/Users/rezasajadian/storage/Training/deep_learning/computer-vision/Object_Tracking_Using_Particle_Filters/ParticleFilterObjectTracking_class/configurations.ini"

#pf = ParticleFilter(config_file)
pf = ParticleFilter(video_file)

pf.run()




