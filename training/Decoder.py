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

    def decode_predictions(self, actionclass_pred, bodypart_pred, offence_pred, touchball_pred, trytoplay_pred):
        # Decode predictions
        actionclass = [self.action_class_map[label.item()] for label in actionclass_pred]
        bodypart = [self.bodypart_map[label.item()] for label in bodypart_pred]
        offence = [self.offence_map[label.item()] for label in offence_pred]
        touchball = [self.touchball_map[label.item()] for label in touchball_pred]
        trytoplay = [self.trytoplay_map[label.item()] for label in trytoplay_pred]

        # Print the decoded outputs
        print("Decoded Predictions:")
        print(f"Action Class: {[x.decode('utf-8') for x in actionclass]}")
        print(f"Body Part: {[x.decode('utf-8') for x in bodypart]}")
        print(f"Offence: {[x.decode('utf-8') for x in offence]}")
        print(f"Touchball: {[x.decode('utf-8') for x in touchball]}")
        print(f"Try to Play: {[x.decode('utf-8') for x in trytoplay]}")