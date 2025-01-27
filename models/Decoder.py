import torch

class Decoder:
    def __init__(self):
        # Define mappings for decoding
        self.action_class_map = {
            0: b'Standing tackling', 1: b'Tackling', 2: b'Holding',
            3: b'Challenge', 4: b'Elbowing', 5: b'High leg',
            6: b'Pushing', 7: b'Dive'
        }
        
        self.bodypart_map = {
            0: b'Under body', 1: b'Use of arms',
            2: b'Use of shoulder', 3: b'Upper body'
        }

        self.offence_map = {
            0: b'No offence', 1: b'Between', 2: b'Offence'
        }
        
        self.touchball_map = {
            0: b'No', 1: b'Maybe', 2: b'Yes'
        }

        self.trytoplay_map = {
            0: b'No', 1: b'Yes'
        }
        
        self.severity_map = {
            0: b'1.0 No card', 1: b'2.0 Borderline No/Yellow', 2: b'3.0 Yellow card', 3: b'4.0 Yellow/ borderline Red', 4: b'5.0 Red card'
        }

    def get_predictions_and_probs(self, predictions):
        """
        Helper method to get the predicted labels and their corresponding probabilities.
        """
        probs = torch.softmax(predictions, dim=1)
        labels = torch.argmax(probs, dim=1)
        max_probs = probs.max(dim=1).values
        return labels, max_probs

    def decode_predictions(self, actionclass_pred, bodypart_pred, offence_pred, touchball_pred, trytoplay_pred, severity_pred):
        """
        Decodes predictions and returns them alongside their probabilities in an array.
        
        Returns:
            list: Array of dictionaries containing decoded predictions and their probabilities
        """
        # Get predictions and probabilities
        actionclass_pred_labels, actionclass_probs = self.get_predictions_and_probs(actionclass_pred)
        bodypart_pred_labels, bodypart_probs = self.get_predictions_and_probs(bodypart_pred)
        offence_pred_labels, offence_probs = self.get_predictions_and_probs(offence_pred)
        touchball_pred_labels, touchball_probs = self.get_predictions_and_probs(touchball_pred)
        trytoplay_pred_labels, trytoplay_probs = self.get_predictions_and_probs(trytoplay_pred)
        severity_pred_labels, severity_probs = self.get_predictions_and_probs(severity_pred)

        # Decode predictions
        actionclass = [self.action_class_map[label.item()] for label in actionclass_pred_labels]
        bodypart = [self.bodypart_map[label.item()] for label in bodypart_pred_labels]
        offence = [self.offence_map[label.item()] for label in offence_pred_labels]
        touchball = [self.touchball_map[label.item()] for label in touchball_pred_labels]
        trytoplay = [self.trytoplay_map[label.item()] for label in trytoplay_pred_labels]
        severity = [self.severity_map[label.item()] for label in severity_pred_labels]

        # Create results array
        results = [
            {
                "category": "Action Class",
                "predictions": [x.decode('utf-8') for x in actionclass],
                "probabilities": actionclass_probs
            },
            {
                "category": "Body Part",
                "predictions": [x.decode('utf-8') for x in bodypart],
                "probabilities": bodypart_probs
            },
            {
                "category": "Offence",
                "predictions": [x.decode('utf-8') for x in offence],
                "probabilities": offence_probs
            },
            {
                "category": "Touchball",
                "predictions": [x.decode('utf-8') for x in touchball],
                "probabilities": touchball_probs
            },
            {
                "category": "Try to Play",
                "predictions": [x.decode('utf-8') for x in trytoplay],
                "probabilities": trytoplay_probs
            },
            {
                "category": "Severity",
                "predictions": [x.decode('utf-8') for x in severity],
                "probabilities": severity_probs
            }
        ]

        return results