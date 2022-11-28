
import numpy as np
import cv2
import os
from configparser import ConfigParser
np.random.seed(0)

class ParticleFilter():
    def __init__(self, config_file):

        # fetching the required parameters from the configuration file
        CONFIGURATIONS = ConfigParser()
        CONFIGURATIONS.read(config_file)

        # video file parameters
        file_name = CONFIGURATIONS["VIDEO_PROPERTIES"]['ADDRESS']
        self.VFILENAME = os.path.join(os.getcwd(), file_name)

        # getting the video properties
        self.video_height_width_getter(self.VFILENAME)

        # Particle parameters
        self.NUM_PARTICLES = int(CONFIGURATIONS['PARTICLE_PARAMETERS']['NUM_PARTICLES'])
        self.VEL_RANGE = float(CONFIGURATIONS['PARTICLE_PARAMETERS']['VEL_RANGE'])

        # Noise parameters
        self.POS_SIGMA = float(CONFIGURATIONS['NOISE_PARAMETERS']['POS_SIGMA'])
        self.VEL_SIGMA = float(CONFIGURATIONS['NOISE_PARAMETERS']['VEL_SIGMA'])

        # self.TARGET_COLOUR = np.array((165, 74, 38))
        self.TARGET_COLOUR = np.array((135, 44, 30))
        # self.TARGET_COLOUR = None
        print(f"Target Color in the beginning :{self.TARGET_COLOUR}")
        
        cv2.namedWindow("Particle Filter for Object Tracking")
        cv2.setMouseCallback('Particle Filter for Object Tracking',self.mouseRGB)
        

    def video_height_width_getter(self, fileName):
        self.video = cv2.VideoCapture(fileName)
        self.VIDEO_WIDTH = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.VIDEO_HEIGHT  = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)


    # reading the video file and yielding frames by OpenCV
    def get_frames(self, fileName):
        self.video = cv2.VideoCapture(fileName)
        while self.video.isOpened():
            global frame
            ret, frame = self.video.read()
            if ret:
                yield frame
            else:
                break
        self.video.release()
        yield None


    def initialize_particles(self):
        particles = np.random.rand(self.NUM_PARTICLES, 4)
        particles = particles * np.array( (self.VIDEO_WIDTH, self.VIDEO_HEIGHT, self.VEL_RANGE, self.VEL_RANGE) )
        # center the Velocity values around zero
        particles[:, -2:] -= self.VEL_RANGE / 2.0
        return particles

    # considering the Velocity feature for the particles
    def apply_velocity(self, particles):
        particles[:, 0] += particles[:, 2]
        particles[:, 1] += particles[:, 3]
        return particles


    # Prevents the particles to fall off the edge of the frames
    def enforce_edges(self, particles):
        for i in range(self.NUM_PARTICLES):
            particles[i, 0] = max(0, min(self.VIDEO_WIDTH - 1, particles[i, 0]))
            particles[i, 1] = max(0, min(self.VIDEO_HEIGHT - 1, particles[i, 1]))
        return particles


    # Measure each particle's quality by comparing its new value to the target value
    def compute_errors(self, particles, frame):
        errors = np.zeros(self.NUM_PARTICLES)
        for i in range(self.NUM_PARTICLES):
            x = int(particles[i, 0])
            y = int(particles[i, 1])
            pixel_colour = frame[y, x, :]
            errors[i] = np.sum( (self.TARGET_COLOUR - pixel_colour) ** 2 )
        return errors

    # giving weights to our particles
    def compute_weights(self, particles, errors):
        weights = np.max(errors) - errors
        weights[
            (particles[:, 0] == 0) |
            (particles[:, 0] == self.VIDEO_WIDTH - 1) |
            (particles[:, 1] == 0) |
            (particles[:, 1] == self.VIDEO_HEIGHT - 1)
        ] == 0.0

        # the power of the weights could go higher, and it heavily affects the
        # behaviour of the particles and their convergence speed, right after initiation.
        weights = weights ** 4
        return weights

    # adding a noise feature to our particel weights
    def apply_noise(self, particles):
        noise = np.concatenate(
            (
                np.random.normal(0.0, self.POS_SIGMA, (self.NUM_PARTICLES, 1)),
                np.random.normal(0.0, self.POS_SIGMA, (self.NUM_PARTICLES, 1)),
                np.random.normal(0.0, self.VEL_SIGMA, (self.NUM_PARTICLES, 1)),
                np.random.normal(0.0, self.VEL_SIGMA, (self.NUM_PARTICLES, 1))
            ),
            axis=1
        )
        particles += noise
        return particles


    def resample(self, particles, weights):
        probabilities = weights / np.sum(weights)
        index_numbers = np.random.choice(self.NUM_PARTICLES, size=self.NUM_PARTICLES, p=probabilities)
        particles = particles[index_numbers, :]

        x = int(np.mean(particles[:, 0]))
        y = int(np.mean(particles[:, 1]))

        return particles, (x, y)


    def display(self, frame, particles, location):
        if self.TARGET_COLOUR is not None:
            if len(particles) > 0:
                for i in range(self.NUM_PARTICLES):
                    x = int(particles[i, 0])
                    y = int(particles[i, 1])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), 1)
                if len(location) > 0:
                    cv2.circle(frame, location, 15, (0, 0, 255), 5)
        cv2.imshow('Particle Filter for Object Tracking', frame)
        if cv2.waitKey(30) == 27:
            if cv2.waitKey(0) == 27:
                return True
        return False


    # gets the click point pixel information for the tracker.
    def mouseRGB(self, event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            colorsB = frame[y,x,0]
            colorsG = frame[y,x,1]
            colorsR = frame[y,x,2]
            self.TARGET_COLOUR = np.array((colorsB, colorsG, colorsR))
            print(f"click color:{self.TARGET_COLOUR}")


    def run(self):
        particles = self.initialize_particles()
        for frame in self.get_frames(self.VFILENAME):
            if frame is None: break
            if self.TARGET_COLOUR is None:
                    self.display(frame, None, None)
            else:
                particles = self.apply_velocity(particles)
                particles = self.enforce_edges(particles)
                errors = self.compute_errors(particles, frame)
                weights = self.compute_weights(particles, errors)
                particles, location = self.resample(particles, weights)
                particles = self.apply_noise(particles)
                terminate = self.display(frame, particles, location)
                if terminate:
                    break


        cv2.destroyAllWindows()
