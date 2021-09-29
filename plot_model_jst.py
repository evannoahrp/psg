from keras.models import load_model
from keras.utils.vis_utils import plot_model
from keras.utils.vis_utils import model_to_dot

#load the saved model
saved_model = load_model('best_model_TBU.h5')
plot_model(saved_model, to_file='model_plot_TBUv1.png', show_shapes = True, show_layer_names = True)
saved_model.summary()