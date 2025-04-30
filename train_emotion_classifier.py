from model import get_model
from prepare_data import get_training_data
import os



X_train, y_train, X_test, y_test = get_training_data()

model = get_model(X_train.shape[1], y_train.shape[1], 1e-6, 'saved_models/Emotion_Model_v2.h5')
model_history=model.fit(X_train, y_train, batch_size=4, epochs=50, validation_data=(X_test, y_test), verbose=1)

# Save model and weights
model_name = 'Emotion_Model_v3.h5'
save_dir = os.path.join(os.getcwd(), 'saved_models')

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Save model and weights at %s ' % model_path)

# Save the model to disk
model_json = model.to_json()
with open("model_json.json", "w") as json_file:
    json_file.write(model_json)