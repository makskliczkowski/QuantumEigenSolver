from ..common.__common__ import *

''' RBM state class '''

class rbmState:
    def __init__(self,
                n_visible,
                n_hidden,
                lr : float = 1e-1,
                momentum : float = 0.95,
                xavier_const : float = 1.0
                ):
        """
        Initializes RBM.
        :param n_visible: number of visible neurons (input size)
        :param n_hidden: number of hidden neurons
        :param learning_rate: learning rate (default: 0.01)
        :param momentum: momentum (default: 0.95)
        :param xavier_const: constant used to initialize weights (default: 1.0)
        """
        if not 0.0 <= momentum <= 1.0:
            raise ValueError('momentum should be in range [0, 1]')
        
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = lr
        self.momentum = momentum
        self.xavier_const = xavier_const
        
    def initialize(self):
        xavier = tf.keras.initializers.GlorotNormal(seed = None)

        self.W = tf.Variable(tf.complex(
                                xavier(shape = (self.n_hidden, self.n_visible)),
                                xavier(shape = (self.n_hidden, self.n_visible)) 
                                    )
                            )
        self.visible_bias = tf.Variable(tf.complex(
                                xavier(shape = (self.n_visible)),
                                xavier(shape = (self.n_visible)) 
                                    )
                            )
        self.hidden_bias = tf.Variable(tf.complex(
                        xavier(shape = (self.n_hidden)),
                        xavier(shape = (self.n_hidden)) 
                            )
                    )
    ''' defines the given coefficient for state s'''
    def p(self, s):
        "Probability amplitude of a visible state `v`. We don't need it for Monte Carlo."
        return tf.math.exp(tf.tensordot(tf.math.conj(self.visible_bias), s)) * tf.reduce_prod(tf.math.cosh(self.c + tf.linalg.matvec(self.W * s))) * (2**self.n_hidden)