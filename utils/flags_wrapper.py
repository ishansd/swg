

class flags_wrapper():
    """flags_wrapper
        My wrapper around hyperparameters. This will be passed to the generative 
        model.
    """

    def __init__(
            self,
            learning_rate=1e-4,
            batch_size=64,
            latent_dim=100,
            max_iters=10000,
            num_projections=10000,
            use_discriminator=True):

        self.learning_rate = learning_rate
        self.batch_size = int(batch_size)
        self.max_iters = int(max_iters)

        # For input noise
        self.latent_dim = int(latent_dim)

        # SWG specific params
        self.num_projections = int(num_projections)
        self.use_discriminator = use_discriminator

        return
