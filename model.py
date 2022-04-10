from sklearn.pipeline  import Pipeline 

class ClassificationPredictor:
    def __init__(self):
        self._pipeline = None
        
    def fit(self, X, y):
        self._pipeline.fit(X, y)

    def add_pipeline_step(self, step):
        """Add a step to the pipeline.
        Args:
            step (tuple): The first element should be the name of the step and the
            second element should be the function to execute.
        Returns:
            None
        """
        if not self._pipeline:
            self._pipeline = Pipeline(steps=[step])
        else:
            self._pipeline.steps.append(step)

    def predict(self, input):
        """Obtain the model's inference from the given input."""
        return self._pipeline.predict(input)