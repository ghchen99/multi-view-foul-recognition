import torch

class Decoder:
    def __init__(self):
        # Define mappings for decoding
        self.action_class_map = {
            int(v): k.encode('utf-8') for k, v in {
                'Standing tackling': '0', 'Tackling': '1', 'Holding': '2',
                'Challenge': '3', 'Elbowing': '4', 'High leg': '5',
                'Pushing': '6', 'Dive': '7'
            }.items()
        }
        
        self.bodypart_map = {
            int(v): k.encode('utf-8') for k, v in {
                'Under body': '0', 'Use of arms': '1',
                'Use of shoulder': '2', 'Upper body': '3'
            }.items()
        }

        self.offence_map = {
            int(v): k.encode('utf-8') for k, v in {
                'No offence': '0', 'Between': '1', 'Offence': '2'
            }.items()
        }
        
        self.touchball_map = {
            int(v): k.encode('utf-8') for k, v in {
                'No': '0', 'Maybe': '1', 'Yes': '2'
            }.items()
        }

        self.trytoplay_map = {
            int(v): k.encode('utf-8') for k, v in {
                'No': '0', 'Yes': '1'
            }.items()
        }
        
        self.severity_map = {
            int(v): f"{float(k):.1f}{' No card' if k == '1.0' else ' Borderline No/Yellow' if k == '2.0' else ' Yellow card' if k == '3.0' else ' Yellow/ borderline Red' if k == '4.0' else ' Red card'}".encode('utf-8')
            for k, v in {
                '1.0': '0', '2.0': '1', '3.0': '2', '4.0': '3', '5.0': '4'
            }.items()
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
                "category": "Touch Ball",
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