# Authentication

*Want to follow along? {download}`Download this notebook.<authentication.md>`*

This document provides information on authentication aspects.
When you install Deltakit, you install a family of Python packages.
While most features are available immediately after installation,
some functions are hosted in the cloud.

This includes:
- adding leakage noise
- leakage noise simulation
- decoding with Ambiguity Clustering (AC), Local Clustering (LC), and Collision Clustering (CC) proprietary decoders
- leakage-aware decoding with the LC Decoder
- iSWAP native gate circuit generation
- generation of quantum stability experiments

To access these features, you will send authenticated requests to the server.
Please generate your authentication token [at this page](https://deltakit.riverlane.com/dashboard/token).

## Basic flow

The basic flow is to persist your token on the file system by executing the below piece of code once.
After you do this, it will automatically be loaded from the file when needed.

```python
from deltakit.explorer import Client
Client.set_token("your_token")
```

Now, even if you restart your Python kernel or reboot your machine, you will be able to build a `Client` class instance:

```python
cloud = Client.get_instance()
```

## Alternative flow

You may prefer not to store your token on the file system.
Instead, you can use the `DELTAKIT_TOKEN` environment variable.
This is useful when you use Deltakit within CI actions in GitHub, or run your code in a temporary environment like Google Colab.

In Python, you can do this using the `os` package:

```python
import os
from deltakit.explorer import Client

os.environ["DELTAKIT_TOKEN"] = "your_token"
cloud = Client.get_instance()
```

If you run a Python script, you may define the variable outside your code.

In a shell:

::::{tab-set}
:::{tab-item} Linux / macOS
:sync: tab1
```bash
DELTAKIT_TOKEN=your_token && python3 some_script.py
```
:::
:::{tab-item} Windows
:sync: tab3
```bash
set "DELTAKIT_TOKEN=your_token" && python3 some_script.py
```
:::
::::

In GitHub CI, you may store a token as a secret and use it in the action's YAML:

```yaml
jobs:
  your-job-name:
    runs-on: ubuntu-latest
      steps:
        - name: Install Deltakit
          run: |
            python3 -m pip install deltakit
        - name: your step name
          run: |
            python3 some_script.py
          env:
            DELTAKIT_TOKEN: ${{ secrets.DELTAKIT_TOKEN }}
```

## Troubleshooting

### Token is not set

If the token is not set, or not provided as an environment variable, usage of the cloud API will raise an exception:

```text
RuntimeError: Token could not be found neither in ([...]/deltakit-explorer/.env) nor environment variable (DELTAKIT_TOKEN). Please obtain your token at https://deltakit.rivelane.com/dashboard/token and use `Client.set_token` function to register it.
```

If you see this message, please visit the [Deltakit token generation page](https://deltakit.rivelane.com/dashboard/token), generate a token, and set it with `Client.set_token(...)`.

### Token is wrong

You may accidentally add or remove a character from the token.
The server will check this and report to you:

```text
ServerException: Token failed validation: Status 401 (Error #6000): Invalid token header. Secret key should be a 32-character string..
```
or
```text
ServerException: Token failed validation: Status 401 (Error #6000): Invalid token header. No credentials provided..
```

Please double-check that you've copied the token fully and use `Client.set_token(...)` to overwrite it.

### Token has expired

Even if the token is correct, you might have regenerated it.
At that moment, the old token becomes invalid.

```text
ServerException: Token failed validation: Status 401 (Error #6000): Invalid token received..
```

Please use only the most recent token.

### Service unavailable

Even if your token is correct, you may occasionally receive a `ServerException` with a "no healthy upstream" message.
This may mean that the server is being maintained, and you should retry your request later.
