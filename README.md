# HelloWorld: Enabling Socially Interactive Characters in Video World Models

Liangyang Ouyang<sup>1,2</sup>, Ruicong Liu<sup>2</sup>, Xuangeng Chu<sup>2</sup>, Kaipeng Zhang<sup>2</sup>, Yoichi Sato<sup>1</sup>

<sup>1</sup>The University of Tokyo &nbsp;&nbsp; <sup>2</sup>Alaya Lab

![Teaser](assets/teaser.png)

**HelloWorld** is a video world model that enables social interaction with in-world characters. With a single button press (`F`), users can prompt the on-screen character to respond toward the camera, *e.g.*, turning to the viewer, waving, nodding, or speaking a short greeting, while maintaining high-quality scene and camera-trajectory reconstruction.

- **Self-distillation training:** the base video generation model is finetuned on data synthesized by itself, containing both social interactions and camera motion, so it learns camera-pose conditioning without degrading interaction quality.
- **Training-free temporal control:** at inference, a temporal cross-attention mask localizes the character's response to the `F`-press window.
- **HelloWorldBench:** a 400-sample benchmark with three social interaction metrics (ActAcc, TimeAcc, GazeDev) alongside three conventional metrics.

## Paper

📄 [HelloWorld.pdf](assets/HelloWorld.pdf)

## Demo Video

[![Watch the demo on YouTube](assets/video_cover.png)](https://youtu.be/j4scl5Y7gXo)

▶️ Watch on [YouTube](https://youtu.be/j4scl5Y7gXo) &nbsp;·&nbsp; 📥 Download: [HelloWorld.mp4](assets/HelloWorld.mp4) (35 MB, 1080p)

## Code & Benchmark

Coming soon.

## Unofficial Reproduction

Many thanks to [ScrappyLabs](https://github.com/scrappylabsai) for independently reproducing HelloWorld and open-sourcing their trained model — see [scrappylabsai/helloworld-interactor](https://github.com/scrappylabsai/helloworld-interactor) and the released LoRA on [Hugging Face](https://huggingface.co/scrappylabsai/helloworld-interactor-lora). We are glad to see that their results align well with the paper. Our official release is in progress; in the meantime, feel free to try their model.

## Contact

For questions, please contact oyly@iis.u-tokyo.ac.jp or liangyang.ouyang@shanda.com.
