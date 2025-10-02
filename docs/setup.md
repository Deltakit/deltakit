# Setup

## Installation

To install the latest version of Deltakit, run:

```
pip install deltakit
```

### Installation in a virtual environment or Colab

In a terminal with your favorite distribution of Python/pip on the path, browse to a working folder for the virtual environment and run:

::::{tab-set}
:::{tab-item} macOS / Linux
:sync: tab1
```bash
# please use your virtual environment,
# or create a new one with these commands:
python3 -m venv venv
source venv/bin/activate

# and then install Deltakit:
pip install deltakit
```
:::
:::{tab-item} Windows
:sync: tab2
```powershell
# please use your virtual environment,
# or create a new one with these commands:
python -m venv venv
venv\Scripts\activate

# and then install deltakit:
pip install deltakit
```

Note: Deltakit depends on `stim` and other libraries that may need to be compiled from source
during installation. If you have trouble compiling, consider installing
[Build Tools for Visual Studio](https://visualstudio.microsoft.com/downloads/)
with the "Desktop development for C++" workload.
:::
:::{tab-item} Colab
:sync: tab3
```bash
# remove conflicting dependency:
!pip uninstall datasets -y
# and then install deltakit:
!pip install deltakit
```
:::
::::

## Authentication
Some Deltakit code relies on requests to servers sent over the internet, so you'll
need to set up credentials before using Deltakit for the first time.

After visiting the [Deltakit website](https://deltakit.riverlane.com/dashboard/token) to
generate an access token, register it with Deltakit.

``` python
from deltakit.explorer import Client

Client.set_token("<your token>")
```
By default, this token is stored in a configuration file, so this setup only needs
to be performed once. If you prefer, you can add your token manually as an
[environment variable](https://www.twilio.com/en-us/blog/how-to-set-environment-variables-html)
`DELTAKIT_TOKEN`.

You can check that your token is correctly configured by running the following code:

```python
client = Client.get_instance()
```

You can provide the resulting ``client`` object as an argument to features that
(currently) require communication with servers.
