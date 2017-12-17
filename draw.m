function draw( frame_ID, index)

input_frame_num = 20
label = readNPY( 'test_gt.npy');
label = squeeze(label(index+1, :, :));

input_frame = readNPY( 'test_input.npy');
input_frame = squeeze(input_frame(index+1, :, :));

predicted_possibility = readNPY(strcat('pred_', int2str(index), '.npy'));

single_pred = readNPY('single_LSTM_pred.npy');
single = squeeze(single_pred(index+1, :, :));
im=imread(strcat(frame_ID, '.jpg'));
imshow(im)
w = 2;
hold on

plot(input_frame(:,1),input_frame(:,2), 'Color', [1, 0.8, 0], 'Marker', 'x','LineWidth',w);

last_input = [input_frame(input_frame_num, 1), input_frame(input_frame_num, 2)]
% label = [last_input];
label = [last_input; label];
plot(label(:,1), label(:,2), 'Color', 'b', 'Marker', 'o','LineWidth',w);



predicted_possibility_1 = squeeze(predicted_possibility(1, :, :));
predicted_possibility_1 = [last_input; predicted_possibility_1];
predicted_possibility_2 = squeeze(predicted_possibility(2, :, :));
predicted_possibility_2 = [last_input; predicted_possibility_2];
predicted_possibility_3 = squeeze(predicted_possibility(3, :, :));
predicted_possibility_3 = [last_input; predicted_possibility_3];


plot(predicted_possibility_1(:,1), predicted_possibility_1(:,2), 'Color', [0, 0.8, 0], 'Marker', '+','LineWidth',w)
plot(predicted_possibility_3(:,1), predicted_possibility_3(:,2), 'Color', 'm', 'Marker', 'p','LineWidth',w)
plot(predicted_possibility_2(:,1), predicted_possibility_2(:,2), 'Color', 'c', 'Marker', 's','LineWidth',w)

single = [last_input; single];
plot(single(:,1), single(:,2), 'Color', 'r', 'Marker', '>','LineWidth',w)



legend('Observed','Ground Truth','93.4%', '3.6%','1.1%', 'without classification')

    


