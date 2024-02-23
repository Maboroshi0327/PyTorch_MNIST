import cv2
import torch
import gradio as gr

from model import CNN_MNIST

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./model.pt"

labels = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
]


def predict(inputs):
    image = inputs["composite"]
    image = cv2.blur(image, (5, 5))
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_LINEAR)
    image = image.reshape(1, 1, 28, 28)
    data = torch.tensor(image).type(torch.FloatTensor) / 255.0
    data = data.to(device)

    model = CNN_MNIST(device=device).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    with torch.no_grad():
        output = model(data)
        output = output.cpu().numpy()[0]
        return {labels[i]: output[i] for i in range(10)}


def main():
    sketchpad = gr.Sketchpad(
        image_mode="L",
        crop_size="1:1",
        brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed"),
    )

    interface = gr.Interface(
        fn=predict,
        inputs=sketchpad,
        outputs="label",
        title="MNIST Sketchpad",
    )
    interface.launch(share=True)


if __name__ == "__main__":
    main()
