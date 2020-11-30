# Object-Detection_YOLO

### Pytorch implementation of YOLO model for object detection project in "IST Seminar at Computer Vision Lab", Kyoto University, Spring 2020.

#### Dataset: 
* Images: size 1x300x300, containing a Hiragana character and Kanji characters. \
* Labels: bounding box and label of the Hiragana character
* [Download](https://drive.google.com/file/d/1vDZMl6iI331XXrNv8JyOY8i3jcjYM48I/view?usp=sharing)

<table>
  <tr>
    <th>
      <img src="https://github.com/phucdoitoan/Object-Detection_YOLO/blob/main/sample.png" width="200" title="Data sample">
    </th>
  </tr>
</table>

#### The model, training and testing is presented in jupyter notebook with check-point model saving for training on google colab's GPUs. But it is straightforward to reorgaize as Python modules and run on local GPU servers.

#### Results:
* Mean IoU: 0.79
* Label accuracy: 0.95

Losses and accuracies in Training and Testing

<table>
  <tr>
    <th>
      <img src="https://github.com/phucdoitoan/Object-Detection_YOLO/blob/main/figures.png" title="losses and accuracies">
    </th>
  </tr>
</table>

Visualization

<table>
  <tr>
    <th>
      <img src="https://github.com/phucdoitoan/Object-Detection_YOLO/blob/main/visualized_results.png" title="visualization">
    </th>
  </tr>
</table>
