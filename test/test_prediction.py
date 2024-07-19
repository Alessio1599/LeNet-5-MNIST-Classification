
# Prediction
    index = 0
    image = test_x[index]
    label = test_y[index]
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    prediction = np.argmax(prediction)
    print('Label: ', label)
    print('Prediction: ', prediction)

    # Show the image
    plt.imshow(image[0, :, :, 0], cmap='gray')
    plt.show()