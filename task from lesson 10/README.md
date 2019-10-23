<div _ngcontent-tcw-c27="" class="homework__info"><div _ngcontent-tcw-c27="" class="homework__info-wrap"><div _ngcontent-tcw-c27="" class="homework__info-text"><!----><div _ngcontent-tcw-c27="" class="homework__number ng-star-inserted"> �������� ������� 8 </div><!----><div _ngcontent-tcw-c27="" class="homework__date"> ���������: 05.08.2019 15:42 </div></div><!----></div><div _ngcontent-tcw-c27="" class="homework__title">������� �� ResNet</div><div _ngcontent-tcw-c27="" class="homework__description wysiwyg-content wysiwyg-content--inverted"><p>������� � �������������� 4 �����������, �������� ������ ������������� CIFAR10</p><p><br></p><p>1) fully connected neural network:</p><p><br></p><ul><li>input _tensor = (b, 3096)</li><li>1024 ReLU</li><li>256 ReLU</li><li>128 ReLU</li><li>10 SoftMax</li></ul><p><br></p><p>2) Convolution + fully connected neural network:</p><p><br></p><ul><li>input _tensor = (b, 32, 32, 3)</li><li>(3, 3, 3, 16) ReLU</li><li>(3, 3, 16, 32) maxpool(2,2) ReLU</li><li>(3, 3, 32, 64, padding=1 for SAME size) maxpool(2,2) ReLU</li><li>tensor transform (b, 7, 7, 64) -&gt; (b, 3136)</li><li>256 ReLU</li><li>128 ReLU</li><li>10 SoftMax;</li></ul><p><br></p><p>3) Deep Convolution + fully connected neural network</p><p><br></p><ul><li>input _tensor = (b, 32, 32, 3)</li><li>(3, 3, 3, 16) ReLU</li><li>(3, 3, 16, 32) maxpool(2,2) ReLU</li><li>(3, 3, 32, 64, padding=1 for SAME size) maxpool(2,2) ReLU</li><li>(3, 3, 64, 64, padding=1 for SAME size) ReLU</li><li>(3, 3, 64, 64, padding=1 for SAME size) ReLU</li><li>(3, 3, 64, 64, padding=1 for SAME size) ReLU</li><li>(3, 3, 64, 64, padding=1 for SAME size) ReLU</li><li>(3, 3, 64, 64, padding=1 for SAME size) ReLU</li><li>(3, 3, 64, 64, padding=1 for SAME size) ReLU</li><li>(3, 3, 64, 64, padding=1 for SAME size) ReLU</li><li>tensor transform (b, 7, 7, 64) -&gt; (b, 3136)</li><li>256 ReLU</li><li>128 ReLU</li><li>10 SoftMax;</li></ul><p><br></p><p>4) Convolution ResNet + fully connected neural network</p><p><br></p><ul><li>input _tensor = (b, 32, 32, 3)</li><li>(3, 3, 3, 16) ReLU</li><li>(3, 3, 16, 32) maxpool(2,2) ReLU</li><li>(3, 3, 32, 64, padding=1 for SAME size) maxpool(2,2) ReLU</li><li>(3, 3, 64, 64, padding=1 for SAME size) ReLU + last output [ResNet]</li><li>(3, 3, 64, 64, padding=1 for SAME size) ReLU + last output [ResNet]</li><li>(3, 3, 64, 64, padding=1 for SAME size) ReLU + last output [ResNet]</li><li>(3, 3, 64, 64, padding=1 for SAME size) ReLU + last output [ResNet]</li><li>(3, 3, 64, 64, padding=1 for SAME size) ReLU + last output [ResNet]</li><li>(3, 3, 64, 64, padding=1 for SAME size) ReLU + last output [ResNet]</li><li>(3, 3, 64, 64, padding=1 for SAME size) ReLU + last output [ResNet]</li><li>tensor transform (b, 7, 7, 64) -&gt; (b, 3136)</li><li>256 ReLU</li><li>128 ReLU</li><li>10 SoftMax;</li></ul><p><br></p><p>�������� �������� � ����� �������� �������. batch size = 32, ��������� ������������� �� ����������.</p></div><div _ngcontent-tcw-c27=""><!----><!----><!----></div></div>