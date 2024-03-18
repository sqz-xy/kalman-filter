import numpy as np

class KalmanFilter:
    position: float = 0
    velocity: float = 0
    
    x: np.array = []
    P: np.array = []
    A: np.array = []
    H: np.array = []
    HT: np.array = []
    R: np.array = []
    Q: np.array = []
    sA: float = 0.1
    sZ: float = 0.1

    def measurement_update(self, update_index, dt): 
        if update_index == 0:
            self.position = 0
            self.velocity = 60
        
        # Add some random noise to the position and velocity
        w = 0
        v = 0
    
        z = self.position + self.velocity * dt + v
        
        self.position = z - v
        self.velocity = 60 + w
        
        return [z, self.position, self.velocity]


    def filter(self, z, update_index, dt):
        # initialise the kalman state
        if update_index == 0: 
            self.x = np.array([[0],[0]])             # State vector for pos and vel
            self.P = np.array([[0, 0],[0, 0]])       # Covariance matrix for x
            self.A = np.array([[1, dt],[0, 1]])      # State transition matrix
            self.H = np.array([[1, 0]])              # State to measurement transition matrix
            self.R = self.sA ** 2                    # Input measurement variance (Sigma A squared)
            
            self.Q = np.array([[(self.R * (dt ** 4)) / 4, (self.R * (dt ** 3)) / 2],
                               [(self.R * (dt ** 3)) / 2,  self.R * (dt ** 2)]])       # Noise covariance
            
        # Predict state
        self.x = self.A.dot(self.x)                                                         
        self.P = self.A.dot(self.P).dot(np.transpose(self.A)) + self.Q
        
        innovation = self.H.dot(self.P).dot(np.transpose(self.H)) + self.R              
        kalman_gain = self.P.dot(np.transpose(self.H)).dot(np.linalg.inv(innovation))

        # Update state
        residual = z - self.H.dot(self.x)
        self.x = self.x + kalman_gain.dot(residual)
        
        # Update covariance
        self.P = self.P - kalman_gain.dot(self.H).dot(self.P)
        
        return [self.x]

if __name__ == '__main__':
    measurement_count = 100
    dt = 0.1
    kalman_filter = KalmanFilter()
    
    for i in range(0, measurement_count):
        z = kalman_filter.measurement_update(i, dt)
        f = kalman_filter.filter(z[0], i, dt)
        print(f) # print update vector
        print('\n')