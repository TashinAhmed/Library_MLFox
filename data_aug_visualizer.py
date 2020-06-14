class PredictionVisualizer:
    
    DARK_BLUE  = (0.1294, 0.5882, 0.9529) # True label (missclassified)
    GREEN      = (0.2980, 0.6863, 0.3137) # Predicted: correct
    RED        = (0.9869, 0.2627, 0.2118) # Predicted: incorrect
    LIGHT_BLUE = (0.7333, 0.8706, 0.9843) # other
    
    def __init__(self):
        """Prediction visualizer.
        
        Usage:
            See the example below in this notebook.
        """
        self.__model = None
        self.__labels = {}  # type: Dict[str, LabelConfig]
        self.__sample = None  # type: Sample
        self.__transform_fn = None  # type: Callable
        self.__transform_fn_steps = None  # type: List[int]
        self.__transofrm_data = []  # type: List[dict]
    
    @property
    def model(self):
        return self.__model
    
    @model.setter
    def model(self, model):
        self.__model = model

    @property
    def transform_fn(self):
        return self.__transform_fn

    def set_transform_fn(self, transform_fn, steps):
        self.__transform_fn = transform_fn
        self.__transform_fn_steps = steps
    
    @property
    def sample(self) -> Sample:
        return self.__sample

    @sample.setter
    def sample(self, sample: Sample):
        self.__sample = sample
    
    def add_label(self, label_config: LabelConfig) -> None:
        self.__labels[label_config['label_id']] = label_config
    
    @property
    def labels(self) -> Dict[str, LabelConfig]:
        return self.__labels
    
    def create(self):
        """Creates the animation."""
        fig, gs, subplots = self.__generate_figure()
        frames = self.__generate_frames()
        outputs = []
        
        for step_index, image in enumerate(frames['images']):
            output = [
                subplots['image'].imshow(image.astype(np.uint8), animated=True, cmap='Greys_r')
            ]
            
            for label_idx, (label_id, label) in enumerate(self.__labels.items()):
                colors = self.__get_colors(label['num_classes'],
                                           self.sample['labels'][label_id],
                                           np.argmax(frames[label_id], axis=1)[step_index]
                                          )
                output.append(
                    subplots[label_id].vlines(np.array([x for x in range(0, label['num_classes'])]),
                                              np.zeros(len(frames)),
                                              frames[label_id][step_index],
                                              colors
                                             )
                )
                

            outputs.append(output)

        return animation.ArtistAnimation(fig, outputs, interval=50, blit=True, repeat=True, repeat_delay=2000)    
    
    def _before_forward(self, transformed_image):
        """Before forward adapter
        
        The default implementation converts the numpy array to torch tensor and adds the missing
        `channel` and `batch` dimensions. You should update this method if your model expects a
        different input format.

        Args:
            transformed_image (numpy.ndarray): Transformed image (rotated, etc), shape: H x W

        Return:
            Prepared images (batch of image 1) for your model. The returned image's shape should
            be the shape of your model's input (For example, Pytorch: B x C x H x W)
        """        
        # Convert to float tensor
        transformed_image = torch.from_numpy(transformed_image).float()
        
        # Add 'channel' dim
        transformed_image = transformed_image.unsqueeze(0)
        
        # Add 'batch' dimension
        transformed_image = transformed_image.unsqueeze(0)
        
        return transformed_image
    
    def _forward(self, input_image):
        """You can make the forward call in here

        Args: 
            input_image (torch.Tensor | any) Prepared input for your model. Shape: B x C x H x W
        
        Return:
            You should return a dictionary of your model's predictions (logits or softmax)
            for every registered labels.
            
            ```
            with torch.no_grad():
                out_graph, out_vowel, out_conso = self.model(input_image)
            
            return {
                'grapheme_root': out_graph,
                'vowel_diacritic': out_vowel,
                'consonant_diacritic': out_conso
            }
            ```

            out_x.shape => B x label.NUM_CLASS
        """
        raise NotImplementedError

    def _softmax(self, outputs):
        """Applies a softmax function and returns the result.

        If your model has a final softmax layer, then you should override this to return
        the `outputs` argument without changes.
        
        The visualizer will call this method for every label separately.
        
        Args:
            outputs (torch.Tensor | any): Your model's output, shape: BATCH x NUM_CLASSES

        Return:
            Softmaxed values
        """
        return F.softmax(outputs, dim=1)

    def _after_forward(self, probabilities):
        """Convert the result to the required format.
        
        Args:
            probabilities (torch.Tensor | any) Your model's output after the `self._softmax` call.
            
        Return: (numpy.ndarray)
        """
        return probabilities.data.cpu().numpy()[0]
    
    def __generate_figure(self):
        """Generates the plot."""
        fig = plt.figure(constrained_layout=True, figsize=(14, 6))
        gs = fig.add_gridspec(len(self.labels), 2)
        
        subplots = {}
        subplots['image'] = fig.add_subplot(gs[:, 0], xticks=[], yticks=[])
        subplots['image'].set_title('Image id: {}'.format(self.sample['image_id']), fontsize=10)

        for label_idx, (label_id, label) in enumerate(self.__labels.items()):
            subplots[label_id] = fig.add_subplot(gs[label_idx, 1], xlim=(-1, label['num_classes']))
            subplots[label_id].set_title('{} (label: {})'.format(label['label_name'], self.sample['labels'][label_id]), fontsize=10)
    
        return fig, gs, subplots
    
    def __generate_frames(self):
        """Generates the frames."""
        
        assert self.model is not None
        assert self.sample is not None
        assert self.transform_fn is not None
        
        h, w = self.sample['image'].shape
        steps = len(self.__transform_fn_steps)
        
        frames = {}
        
        # Placeholder for the transformed images
        frames['images'] = np.zeros((steps, h, w))
        
        # Create placeholders for the labels
        for label_idx, (label_id, label) in enumerate(self.__labels.items()):
            frames[label_id] = np.zeros((steps, label['num_classes']))
            
        for step, transform_step_value in enumerate(self.__transform_fn_steps):
            
            # Transform the original image
            transformed_image = self.__transform_fn(self.sample['image'], transform_step_value)
            
            # Save the transformed image as a new frame
            frames['images'][step, ...] = transformed_image
            
            # Prepare the image for the model
            input_image = self._before_forward(transformed_image.copy())
            
            # Predict
            model_output = self._forward(input_image)
            
            # Add the results to the frames
            for label_id, output_logits in model_output.items():
                frames[label_id][step, ...] = self._after_forward(self._softmax(output_logits))
                
        return frames

    def __get_colors(self, size, target, pred):
        """Generates the colors of the vlines."""
        gra_color = [self.LIGHT_BLUE for _ in range(size)]

        if pred == target:
            gra_color[pred] = self.GREEN
        else:
            gra_color[pred] = self.RED
            gra_color[target] = self.DARK_BLUE

        return gra_color    

class MyVisualizer(PredictionVisualizer):
     def __init__(self):
        super().__init__()

    # The implementation of the `_forward` method is required.
    def _forward(self, input_image):

        if torch.cuda.is_available():
            input_image = input_image.cuda()
        
        with torch.no_grad():
            out_graph, out_vowel, out_conso = self.model(input_image)

        return {
            'grapheme_root': out_graph,
            'vowel_diacritic': out_vowel,
            'consonant_diacritic': out_conso
        }
        
        
    # Implementation below this is optional
    def _before_forward(self, transformed_image):
        """Before forward adapter
        
        The default implementation converts the numpy array to torch tensor and adds the missing
        `channel` and `batch` dimensions. You should update this method if your model expects a
        different input format.

        Args:
            transformed_image (numpy.ndarray): Transformed image (rotated, etc), shape: H x W

        Return:
            Prepared images (a one element batch) for your model. The returned image's shape should
            be the shape of your model's input (For example, Pytorch: B x C x H x W)
        """
        return super()._before_forward(transformed_image)

    def _softmax(self, outputs):
        """Applies a softmax function and returns the result.

        If your model has a final softmax layer, then this method should return
        the `outputs` argument without changes.
        
        The visualizer will call this method for every label separately.
        
        Args:
            outputs (torch.Tensor | any): Your model's output, shape: BATCH x NUM_CLASSES

        Return:
            Softmaxed values
        """
        return super()._softmax(outputs)
    
    def _after_forward(self, probabilities):
        """Convert the result to the required format.
        
        Args:
            probabilities (torch.Tensor | any) Your model's output after the `self._softmax` call.
            
        Return: (numpy.ndarray)
        """
        return super()._after_forward(probabilities)



# Set model
visualizer.model = model

# Add labels to visualizer
visualizer.add_label({'label_id': 'grapheme_root', 'label_name': 'Grapheme Root', 'num_classes': 168})
visualizer.add_label({'label_id': 'vowel_diacritic', 'label_name': 'Vowel Diacritic', 'num_classes': 11})
visualizer.add_label({'label_id': 'consonant_diacritic', 'label_name': 'Consonant Diacritic', 'num_classes': 7})

# Create a new sample
sample: Sample = {
    #  preprocess the image for the model
    'image': crop_resize(parq[bengali_sample_id], size=TARGET_SIZE, pad=PADDING),
    
    # Image id
    'image_id': 'Test_{}'.format(bengali_sample_id),

    # True labels
    'labels': {
        'grapheme_root': bengali_sample['grapheme_root'],
        'vowel_diacritic': bengali_sample['vowel_diacritic'],
        'consonant_diacritic': bengali_sample['consonant_diacritic']
    }
}
    
# Set the new sample
visualizer.sample = sample

# Set rotation.
visualizer.set_transform_fn(rotate_image, get_range(1, -45, 45, 1))
# visualizer.set_transform_fn(scale_image, get_range(1, 0.25, 1.5, 0.02))

# Create animation
anim = visualizer.create()

# Show the JS animation
HTML(anim.to_jshtml())
