![alt text](https://i.imgur.com/kXEbZ8J.png)

[![Accolade](https://img.shields.io/badge/State_of_the_Art-Data--Efficient_RL-blue.svg??style=flat&logo=google-analytics)]()<br>
[![Accolade](https://img.shields.io/badge/Modern,_Academic--Standard-Image_Classification-blue.svg??style=flat&logo=instatus)]()<br>
[![Accolade](https://img.shields.io/badge/Technically--Working-Generative_Modeling-blue.svg??style=flat&logo=angellist)]()<br>
[![Accolade](https://img.shields.io/badge/In--Progress-Multi_Modalities-red.svg??style=flat&logo=laragon)]()<br>
[![Accolade](https://img.shields.io/badge/Unified_in_one_Framework-Seamless,_General.svg??style=flat&logo=immer)]()

### Quick Links

- [Setup](#wrench-setting-up)

- [Examples](#mag-sample-scripts)

- [Agents and performances](#bar_chart-agents--performances)

# :runner: Running The Code

To start a train session, once [installed](#wrench-setting-up):

```console
python Run.py
```

Defaults:

```Agent=Agents.DQNAgent```

```task=atari/pong```

Plots, logs, generated images, and videos are automatically stored in: ```./Benchmarking```.

![alt text](https://i.imgur.com/2jhOPib.gif)

Welcome ye, weary Traveller.

>Stop here and rest at our local tavern,
>
> Where all your reinforcements and supervisions be served, à la carte!

Drink up! :beers:

# :pen: Paper & Citing

For detailed documentation, [see our :scroll:](https://arxiv.com).

[comment]: <> ([![arXiv]&#40;https://img.shields.io/badge/arXiv-<NUMBER>.<NUMBER>-b31b1b.svg?style=flat&#41;]&#40;https://arxiv.org/abs/<NUMBER>.<NUMBER>&#41;)

[comment]: <> (```)

[comment]: <> (@inproceedings{cool,)

[comment]: <> (  title={bla},)

[comment]: <> (  author={Sam Lerman and Chenliang Xu},)

[comment]: <> (  booktitle={bla},)

[comment]: <> (  year={2022},)

[comment]: <> (  url={https://openreview.net})

[comment]: <> (})

[comment]: <> (```)

```bibtex
@article{cool,
  title   = {UnifiedML: A Unified Framework For Intelligence Training},
  author  = {Lerman, Sam and Xu, Chenliang},
  journal = {arXiv preprint arXiv:2203.08913},
  year    = {2022}
}
```

If you use this work, please give us a star :star: and cite the above.

An acknowledgment to [Denis Yarats](https://github.com/denisyarats), whose excellent [DrQV2 repo](https://github.com/facebookresearch/drqv2) inspired much of this library and its design.

# :open_umbrella: Unified Learning?

Yes.

All agents support discrete and continuous control, classification, and generative modeling in a seamless, unified integration of all.

See example scripts of various configurations [below](#mag-sample-scripts).

# :wrench: Setting Up

Let's get to business.

## 1. Clone The Repo

```console
git clone git@github.com:agi-init/UnifiedML.git
cd UnifiedML
```

## 2. Gemme Some Dependencies

```console
conda env create --name ML --file=Conda.yml
```

## 3. Activate Your Conda Env.

```console
conda activate ML
```

Optionally, for GPU support, install Pytorch with CUDA from https://pytorch.org/get-started/locally/.

# :joystick: Installing The Suites

## 1. Classify

Comes preinstalled.

## 2. Atari Arcade

You can use ```AutoROM``` if you accept the license.

```console
pip install autorom
AutoROM --accept-license
```
Then:
```console
mkdir ./Datasets/Suites/Atari_ROMS
AutoROM --install-dir ./Datasets/Suites/Atari_ROMS
ale-import-roms ./Datasets/Suites/Atari_ROMS
```

## 3. DeepMind Control

Download MuJoCo from here: https://mujoco.org/download.

Make a ```.mujoco``` folder in your home directory:

```console
mkdir ~/.mujoco
```

Extract and move downloaded MuJoCo folder into ```~/.mujoco```. For a linux x86_64 architecture, this looks like:

```console
tar -xf mujoco210-linux-x86_64.tar.gz
mv mujoco210/ ~/.mujoco/ 
```

And run:

```console
pip install --user dm_control
```

to install DeepMind Control. For any issues, consult the [DMC repo](https://github.com/deepmind/dm_control).

# :file_cabinet: Key files

```Run.py``` handles training and evaluation loops, saving, distributed training, logging, plotting.

```Environment.py``` handles rollouts.

```./Agents``` contains self-contained agents.

# :mag: Sample scripts

### RL

Humanoid example:
```console
python Run.py task=dmc/humanoid_run
```

DrQV2 Agent in Atari:
```console
python Run.py Agent=Agents.DrQV2Agent task=atari/battlezone
```

SPR Agent in DeepMind Control:
```console
python Run.py Agent=Agents.SPRAgent task=dmc/humanoid_walk
```

### Classification

DQN Agent on MNIST:

```console
python Run.py task=classify/mnist RL=false
```

*Note:* ```RL=false``` sets training to standard supervised-only classification. Without ```RL=false```, an additional RL update joins the supervised learning update s.t. ```reward = -error```.

Alternatively, and interestingly, ```supervise=false``` will *only* supervise via RL ```reward = -error``` (**experimental**). This is pure RL training and actually works.

[comment]: <> (The latent optimization could also be done over a learned parameter space as in POPLIN &#40;Wang and Ba, 2019&#41;, which lifts the domain of the optimization problem eq. &#40;1&#41; from Y to the parameter space of a fully-amortized neural network. This leverages the insight that the parameter space of over-parameterized neural networks can induce easier non-convex optimization problems than in the original space, which is also studied in Hoyer et al. &#40;2019&#41;.)

Train accuracies can be printed with ```agent.log=true```.

Evaluation with exponential moving average (EMA) of params can be toggled with the ```ema=true``` flag. See [Custom Architectures](#custom-architectures) for mix-and-matching pre-defined architectures via command line. Training with weight decay can be toggled via ```weight_decay=``` and torchvision transforms can be passed in as dicts via ```replay.transform=```. For example,

```console
python Run.py task=classify/cifar10 RL=false ema=true weight_decay=0.01 replay.transform="{RandomHorizontalFlip:{}}" Eyes=Blocks.Architectures.ResNet18
```

[comment]: <> (Rollouts fill up data in an online fashion, piecemeal, until depletion &#40;all data is processed&#41; and gather metadata like past predictions, which may be useful for curriculum learning.)

[comment]: <> (Automatically toggles ```offline=true``` by default, but can be set to ```false``` if past predictions or "streaming" data is needed.)

### Generative Modeling

Via the ```generate=true``` flag:
```console
python Run.py task=classify/mnist generate=true
```
Implicitly treats as offline, and assumes a replay [is saved](#saving) that can be loaded.

Can also work with RL (due to frame stack, the generated images are technically multi-frame videos), but make sure to change some of the default settings to speed up training, as per below:

```console
python Run.py task=atari/breakout generate=true evaluate_episodes=1 action_repeat=1 
```

[comment]: <> (ensemble could help this:)

[comment]: <> (Extensions. Analyzing and extending the amortization components has been a key development in AVI methods. Cremer et al. &#40;2018&#41; investigate suboptimality in these models are categorize it as coming from an amortization gap where the amortized model for eq. &#40;30&#41; does not properly solve it, or the approximation gap where the variational posterior is incapable of approximating the true distribution. Semi-amortization plays a crucial role in addressing the amortization gap and is explored in the semi-amortized VAE &#40;SAVAE&#41; by)

[comment]: <> (Kim et al. &#40;2018&#41; and iterative VAE &#40;IVAE&#41; by Marino et al. &#40;2018&#41;.)

### Offline RL

From a saved experience replay, sans additional rollouts:

```console
python Run.py task=atari/breakout offline=true
```

Assumes a replay [is saved](#saving).

Implicitly treats ```replay.load=true``` and ```replay.save=true```, and only does evaluation rollouts.

Is true by default for classification; replays are automatically downloaded.

### Saving

**Agents** can be saved periodically or loaded with the ```save_per_steps=``` or ```load=true``` flags, and are automatically saved at end of training with ```save=true``` by default.

```console
python Run.py save_per_steps=100000 load=true
```

An **experience replay** can be saved or loaded with the ```replay.save=true``` or ```replay.load=true``` flags.

```console
python Run.py replay.save=true replay.load=true
```

Agents and replays save to ```./Checkpoints``` and ```./Datasets/ReplayBuffer``` respectively per a unique experiment.

Careful, without ```replay.save=true``` a replay, whether new or loaded, will be deleted upon terminate, except for the default offline classification replays.

Replays also save uniquely w.r.t. a date-time to allow multiple independent runs at the same time. In case of multiple saved replays per a unique experiment, the most recent is loaded with `load=true`.

### Custom Architectures

One can also optionally pass in custom architectures such as those defined in ```./Blocks/Architectures```.

ResNet18 on CIFAR-10:

```console
python Run.py task=classify/cifar10 Eyes=Blocks.Architectures.ResNet18 
```

Atari with ViT:

```console
python Run.py Eyes=Blocks.Architectures.ViT +recipes.encoder.eyes.patch_size=7
```

Shorthands like ```Eyes``` and ```pool``` make it easy to plug and play custom architectures, but all of an agent's architectural parts can be accessed, mixed, and matched with the ```recipes.``` keyword.

<details>
<summary><i>See more examples :open_book: </i></summary>
<br>

CIFAR-10 with ViT:

```console
python Run.py Eyes=Blocks.Architectures.ViT task=classify/cifar10 RL=false ema=true weight_decay=0.01 +recipes.encoder.eyes.depth=6 +recipes.encoder.eyes.out_channels=512 +recipes.encoder.eyes.hidden_dim=512 replay.transform="{RandomCrop:{size:32,padding:4},RandomHorizontalFlip:{}}" recipes.Aug=Blocks.Architectures.Null
```

A GAN with a CNN Discriminator:

```console
python Run.py generate=True recipes.critic.q_head._target_=Blocks.Architectures.CNN recipes.critic.q_head.input_shape='${obs_shape}' 
```

Here is a more complex example, disabling the Encoder's flattening of the feature map, and instead giving the Actor and Critic unique Attention Pooling operations on their trunks to pool the unflattened features. The ```Null``` architecture disables that flattening component,

```console
python Run.py task=classify/mnist recipes.critic.trunk._target_=Blocks.Architectures.AttentionPool recipes.actor.trunk._target_=Blocks.Architectures.AttentionPool pool=Blocks.Architectures.Null
```

since otherwise ```repr_shape``` is flattened, with no features for the attention to pool.

[comment]: <> (<details>)

[comment]: <> (<summary><i>See even more examples :open_book: </i></summary>)

[comment]: <> (<br>)

[comment]: <> (Here's how you can load another saved agent's encoder from a pre-configured agent checkpoint ```<path>```:)

[comment]: <> (```)

[comment]: <> (python Run.py Eyes=Utils.load +recipes.encoder.eyes.path=<path> +recipes.encoder.eyes.attr=encoder.Eyes)

[comment]: <> (```)

[comment]: <> (You can imagine training a GAN CNN and then seamlessly using it for RL.)

[comment]: <> (<br>)

[comment]: <> (</details>)
</details>

Of course, it's always possible to just modify the code itself, which may be easier. See for example the two CNN variants in ```./Blocks/Encoders.py```.

[comment]: <> (</details>)

### Distributed

The simplest way to do distributed training is to use the ```parallel=true``` flag,

```console
python Run.py parallel=true 
```

which automatically parallelizes the Encoder's "Eyes" across all visible GPUs. The Encoder is usually the most compute-intensive architectural portion.

To share whole agents across multiple parallel instances,
<details>
<summary><i>Click to open :open_book: </i></summary>
<br>

you can use the ```load_per_steps=``` flag.

For example, a data-collector agent and an update agent,

```console
python Run.py learn_per_steps=0 replay.save=true load_per_steps=1 
```

```console
python Run.py offline=true save_per_steps=2
```

in concurrent processes.

Since both use the same experiment name, they will save and load from the same agent and replay, thereby emulating distributed training. **Highly experimental!**

The order in which these are run matters, lest the replays be saved to and loaded from different paths, potentially.
</details>

### Experiment naming, plotting

The ```experiment=``` flag can help differentiate a distinct experiment; you can optionally control which experiment data is automatically plotted with ```plotting.plot_experiments=```.

```console
python Run.py experiment=ExpName1 plotting.plot_experiments="['ExpName1', 'SomeOtherExp']"
```

A unique experiment for benchmarking and saving purposes, is distinguished by: ```experiment=```, ```Agent=```, ```task=```, and ```seed=``` flags.

# :bar_chart: Agents & Performances

# :interrobang: How is this possible

We use our new Creator framework to unify RL discrete and continuous action spaces, as elaborated in our [paper](https://arxiv.com).

Then we frame actions as "predictions" in supervised learning. We can even augment supervised learning with an RL phase, treating reward as negative error.

For generative modeling, well, it turns out that the difference between a Generator-Discriminator and Actor-Critic is rather nominal.

# :mortar_board: Pedagogy and Research

All files are designed to be useful for educational and innovational purposes in their simplicity and structure.

# Contributing

Support financially:  
[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg?style=flat)](https://www.paypal.com/cgi-bin/)

We are a nonprofit, single-PhD student team whose bank account is quickly hemmoraging.

Feel free to [contact **agi.\_\_init\_\_**](mailto:agi.init@gmail.com). 

Angel investors welcome.

# Note

### If you are only interested in the RL portion,

Check out our [**UnifiedRL**](https:github.com/agi-init/UnifiedRL) library.

It does with RL to this library what PyCharm does with Python to IntelliJ, i.e., waters it down mildly and rebrands a little.~

<hr class="solid">

[MIT license Included.](MIT_LICENSE)