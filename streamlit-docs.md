### Run Streamlit Hello App

Source: https://docs.streamlit.io/get-started/installation/command-line

Launch the built-in Streamlit 'Hello' example application to verify the installation. This command opens the app in your web browser.

```bash
streamlit hello

# Or the long-form command:
python -m streamlit hello
```

--------------------------------

### Install pytest for Streamlit

Source: https://docs.streamlit.io/develop/concepts/app-testing/get-started

Installs the pytest framework, a common Python testing tool, into your Streamlit development environment. This is a prerequisite for running the examples in this guide.

```bash
pip install pytest
```

--------------------------------

### Installation and Basic Usage

Source: https://docs.streamlit.io/develop/quick-reference/cheat-sheet

Instructions for installing Streamlit, running an application, and the standard import convention.

```APIDOC
## Installation and Basic Usage

### Description
Install Streamlit using pip and run your Streamlit application.

### Method
Command Line

### Endpoint
N/A

### Parameters
None

### Request Example
```bash
pip install streamlit
streamlit run first_app.py
```

### Response
N/A

### Import Convention
```python
import streamlit as st
```
```

--------------------------------

### Create a 'Hello World' Streamlit App

Source: https://docs.streamlit.io/get-started/installation/anaconda-distribution

A basic Streamlit application that displays 'Hello World' using the `st.write` function. This serves as a starting point for building Streamlit apps.

```python
import streamlit as st

st.write("Hello World")
```

--------------------------------

### Install and Run Streamlit - Terminal

Source: https://docs.streamlit.io/get-started/installation_slug=%28&slug=develop&slug=concepts&slug=configuration

This snippet demonstrates the basic commands to install Streamlit using pip and then run the 'streamlit hello' command to verify the installation. This is suitable for experienced Python developers familiar with terminal usage.

```bash
pip install streamlit
streamlit hello
```

--------------------------------

### Install and Run Streamlit Component Templates (Shell)

Source: https://docs.streamlit.io/develop/concepts/custom-components/intro

Commands to install npm dependencies and start the Vite dev server for the Reactless TypeScript template, and to activate Python environments and run Streamlit apps for both React and Reactless templates.

```shell
npm install
npm run start
```

```shell
# React template
cd template
. venv/bin/activate # or similar to activate the venv/conda environment where Streamlit is installed
pip install -e . # install template as editable package
streamlit run my_component/example.py # run the example

# or

# TypeScript-only template
cd template-reactless
. venv/bin/activate # or similar to activate the venv/conda environment where Streamlit is installed
pip install -e . # install template as editable package
streamlit run my_component/example.py # run the example
```

--------------------------------

### Initialize and Run Streamlit App Test

Source: https://docs.streamlit.io/develop/concepts/app-testing/get-started

Initializes a Streamlit app test using `AppTest.from_file()` and immediately runs the app to capture its initial state. This is the first step in simulating user interactions within a test.

```python
at = AppTest.from_file("app.py").run()
```

--------------------------------

### Setup Connection

Source: https://docs.streamlit.io/develop/api-reference/connections

Demonstrates how to establish a connection to a data source or API using `st.connection` and query data.

```APIDOC
## POST /websites/streamlit_io/connections/setup

### Description
Connect to a data source or API using Streamlit's connection manager.

### Method
POST

### Endpoint
`/websites/streamlit_io/connections/setup`

### Parameters
#### Query Parameters
- **name** (string) - Required - The name of the connection.
- **type** (string) - Required - The type of connection (e.g., 'sql', 'snowflake').

### Request Body
This endpoint does not typically require a request body for basic setup, but custom connections might.

### Request Example
```python
conn = st.connection('pets_db', type='sql')
pet_owners = conn.query('select * from pet_owners')
st.dataframe(pet_owners)
```

### Response
#### Success Response (200)
Returns a connection object that can be used to interact with the data source.

#### Response Example
```json
{
  "message": "Connection established successfully"
}
```
```

--------------------------------

### Install apt-get Dependencies with packages.txt

Source: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies

This snippet demonstrates the content of a `packages.txt` file used to install system dependencies required by Python packages like `mysqlclient`. The file should list one package name per line. These packages are installed using `apt-get` on Debian Linux.

```text
build-essential
pkg-config
default-libmysqlclient-dev
```

--------------------------------

### Dockerfile for Streamlit Deployment

Source: https://docs.streamlit.io/deploy/tutorials/kubernetes

A Dockerfile that defines the steps to build a Docker image for a Streamlit application. It installs dependencies, clones an example Streamlit app, sets up a virtual environment, and configures the entrypoint to run the application.

```dockerfile
FROM python:3.9-slim

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 -ms /bin/bash appuser

RUN pip3 install --no-cache-dir --upgrade \
    pip \
    virtualenv

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git

USER appuser
WORKDIR /home/appuser

RUN git clone https://github.com/streamlit/streamlit-example.git app

ENV VIRTUAL_ENV=/home/appuser/venv
RUN virtualenv ${VIRTUAL_ENV}
RUN . ${VIRTUAL_ENV}/bin/activate && pip install -r app/requirements.txt

EXPOSE 8501

COPY run.sh /home/appuser
ENTRYPOINT ["./run.sh"]

```

--------------------------------

### Install Python Packages for Streamlit and Neon

Source: https://docs.streamlit.io/develop/tutorials/databases/neon

Installs essential Python packages for Streamlit development and PostgreSQL connectivity. Ensure these are in your environment before proceeding.

```txt
streamlit>=1.28
psycopg2-binary>=2.9.6
sqlalchemy>=2.0.0
```

--------------------------------

### Run a Streamlit App

Source: https://docs.streamlit.io/get-started/installation/anaconda-distribution

Executes a Streamlit application file from the terminal. This command starts a local web server and opens the app in your default browser.

```bash
streamlit run app.py
```

```bash
python -m streamlit run app.py
```

--------------------------------

### Initialize and Run Streamlit App for Testing (Python)

Source: https://docs.streamlit.io/develop/concepts/app-testing/get-started

Initializes a Streamlit app from a Python file and runs it for testing purposes. This is the primary method for setting up a simulated app environment. It returns an `AppTest` instance that can be further manipulated.

```python
# Path to app file is relative to myproject/
at = AppTest.from_file("app.py").run()
```

```python
"""test_app.py"""
from streamlit.testing.v1 import AppTest

def test_increment_and_add():
    """A user increments the number input, then clicks Add"""
    at = AppTest.from_file("app.py").run()
    at.number_input[0].increment().run()
    at.button[0].click().run()
    assert at.markdown[0].value == "Beans counted: 1"
```

```python
# Initialize the app.
at = AppTest.from_file("app.py")
# Run the app.
at.run()
```

--------------------------------

### Example config.toml Structure

Source: https://docs.streamlit.io/develop/api-reference/configuration/config

A basic example demonstrating the structure of a config.toml file, including client and theme settings. This file is used to customize Streamlit application behavior and appearance.

```toml
[client]
showErrorDetails = "none"

[theme]
primaryColor = "#F63366"
backgroundColor = "black"
```

--------------------------------

### Streamlit CLI: Example App

Source: https://docs.streamlit.io/develop/api-reference/cli

Command to run a pre-built example Streamlit application. This is useful for quick testing or demonstration.

```bash
streamlit hello
```

--------------------------------

### Navigate to Project Directory

Source: https://docs.streamlit.io/get-started/installation/command-line

Change the current directory to your project folder. This is the first step before creating a virtual environment.

```bash
cd myproject
```

--------------------------------

### Streamlit Counter App with Callbacks and Args

Source: https://docs.streamlit.io/develop/concepts/architecture/session-state

This Streamlit example enhances the counter app by allowing users to specify the increment value using a number input. It demonstrates passing arguments to a callback function using the 'args' parameter of a button widget.

```python
import streamlit as st

st.title('Counter Example using Callbacks with args')
if 'count' not in st.session_state:
    st.session_state.count = 0

increment_value = st.number_input('Enter a value', value=0, step=1)

def increment_counter(increment_value):
    st.session_state.count += increment_value

increment = st.button('Increment', on_click=increment_counter,
    args=(increment_value, ))

st.write('Count = ', st.session_state.count)
```

--------------------------------

### Validate Streamlit Installation

Source: https://docs.streamlit.io/get-started/installation/anaconda-distribution

Verifies that Streamlit has been installed correctly by running the built-in 'hello' command. This command launches a sample Streamlit app in your browser.

```bash
streamlit hello
```

```bash
python -m streamlit hello
```

--------------------------------

### Install Streamlit Component from Test PyPI

Source: https://docs.streamlit.io/develop/concepts/custom-components/publish

This command installs a Streamlit Component package from the Test PyPI repository into a Python environment. It's used to verify that the component was uploaded correctly and can be installed by users. The `--index-url` flag specifies Test PyPI, and `--no-deps` prevents automatic installation of dependencies.

```bash
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps example-pkg-YOUR-USERNAME-HERE
```

--------------------------------

### Run Docker Hello World

Source: https://docs.streamlit.io/deploy/tutorials/kubernetes

Verifies the correct installation of Docker Engine by running the 'hello-world' Docker image. This is a fundamental step to ensure Docker is operational on your system.

```bash
sudo docker run hello-world
```

--------------------------------

### Create a 'Hello World' Streamlit App

Source: https://docs.streamlit.io/get-started/installation/command-line

Create a Python file named 'app.py' containing a simple Streamlit script that displays 'Hello world'. This serves as a basic Streamlit application.

```python
import streamlit as st

st.write("Hello world")
```

--------------------------------

### Start a Simple HTTP Server (Python Terminal)

Source: https://docs.streamlit.io/knowledge-base/deploy/remote-start

This command starts a basic HTTP server using Python's built-in module. It's used to test if the server environment can serve content, helping to isolate whether issues are Streamlit-specific or environmental.

```bash
python -m http.server [port]
```

--------------------------------

### Install Git and Build Tools in Docker

Source: https://docs.streamlit.io/deploy/tutorials/docker

Installs essential build tools, curl, software properties common, and git within a Docker container. It also cleans up apt lists to reduce image size. This is a prerequisite for cloning code from remote repositories.

```dockerfile
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*
```

--------------------------------

### Streamlit: Implementing Top Navigation Menu

Source: https://docs.streamlit.io/develop/api-reference/navigation/st

This example shows how to implement a top navigation menu in a Streamlit application using the `position` parameter. This approach is beneficial for apps with numerous pages, enabling collapsible sections for better organization. It uses the same page grouping structure as the previous example.

```Python
import streamlit as st

pages = {
    "Your account": [
        st.Page("create_account.py", title="Create your account"),
        st.Page("manage_account.py", title="Manage your account"),
    ],
    "Resources": [
        st.Page("learn.py", title="Learn about us"),
        st.Page("trial.py", title="Try it out"),
    ],
}

pg = st.navigation(pages, position="top")
pg.run()
```

--------------------------------

### Install and Run Streamlit Applications

Source: https://docs.streamlit.io/develop/quick-reference/cheat-sheet

Instructions for installing Streamlit using pip and running a Streamlit application. It also shows the conventional way to import the Streamlit library in Python scripts.

```python
pip install streamlit

streamlit run first_app.py

# Import convention
>>> import streamlit as st
```

--------------------------------

### Run Custom Streamlit App

Source: https://docs.streamlit.io/get-started/installation/command-line

Execute your custom Streamlit application ('app.py') using the 'streamlit run' command. Ensure your virtual environment is activated first.

```bash
streamlit run app.py

# Or the long-form command:
python -m streamlit run app.py
```

--------------------------------

### Using Magic Commands for Help in Streamlit

Source: https://docs.streamlit.io/develop/api-reference/utilities/st

This example showcases Streamlit's 'Magic' feature, allowing users to get help for functions, classes, or modules by simply referencing them without explicitly calling st.help. This provides a more interactive way to explore Streamlit and other libraries.

```python
import streamlit as st
import pandas

# Get help for Pandas read_csv:
pandas.read_csv

# Get help for Streamlit itself:
st
```

--------------------------------

### Run Streamlit App (GitHub Repo)

Source: https://docs.streamlit.io/develop/api-reference/cli/run

Executes a Streamlit app directly from a public GitHub repository or gist. This is convenient for sharing and running examples without local setup.

```bash
streamlit run https://raw.githubusercontent.com/streamlit/demo-uber-nyc-pickups/master/streamlit_app.py
```

--------------------------------

### Create MySQL Database and Table

Source: https://docs.streamlit.io/develop/tutorials/databases/mysql

SQL commands to create a 'pets' database, a 'mytable' table within it, and insert sample data. This is a prerequisite for the Streamlit connection example.

```sql
CREATE DATABASE pets;

USE pets;

CREATE TABLE mytable (
    name varchar(80),
    pet varchar(80)
);

INSERT INTO mytable VALUES ('Mary', 'dog'), ('John', 'cat'), ('Robert', 'bird');
```

--------------------------------

### Streamlit: Basic Page Navigation with Files and Callables

Source: https://docs.streamlit.io/develop/api-reference/navigation/st

This example demonstrates how to set up basic navigation in a Streamlit app. It shows how to include Python files or callable functions as pages within the navigation. Page titles, icons, and paths are automatically inferred from the file or callable names.

```Python
import streamlit as st

def page_2():
    st.title("Page 2")

pg = st.navigation(["page_1.py", page_2])
pg.run()
```

--------------------------------

### Retrieve Streamlit Elements by Index (Python)

Source: https://docs.streamlit.io/develop/concepts/app-testing/get-started

Demonstrates how to access specific Streamlit elements (like buttons or markdown) within a simulated app using their index. Elements are ordered by their appearance on the page.

```python
import streamlit as st

first = st.container()
second = st.container()

second.button("A")
first.button("B")
```

```python
assert at.button[0].label == "B"
assert at.button[1].label == "A"
```

--------------------------------

### Initialize Streamlit App with Imports

Source: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/validate-and-edit-chat-responses

This Python code snippet shows the necessary imports to start building a Streamlit application. It includes 'streamlit' for the web framework, 'lorem' for generating text, 'random' for randomness, and 'time' for delays. These are foundational for creating interactive and dynamic web apps.

```python
import streamlit as st
import lorem
from random import randint
import time
```

--------------------------------

### Configure Streamlit App CI Workflow with GitHub Actions

Source: https://docs.streamlit.io/develop/concepts/app-testing/examples

This workflow automates the deployment of a Streamlit application. It checks out the code, sets up Python, and uses the streamlit-app-action to run the application. It is triggered on push and pull request events to the 'main' branch.

```yaml
name: Streamlit app

on:
  push:
    branches: ["main"]
  pull_request:
    branches: ["main"]

permissions:
  contents: read

jobs:
  streamlit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - uses: streamlit/streamlit-app-action@v0.0.3
        with:
          app-path: app.py
```

--------------------------------

### Install SQLAlchemy for Streamlit SQL Connections

Source: https://docs.streamlit.io/develop/concepts/connections/connecting-to-data

This command installs the SQLAlchemy library, which is required for Streamlit's SQL connection functionality. Ensure you use the specified version for compatibility.

```bash
pip install SQLAlchemy==1.4.0
```

--------------------------------

### Upgrade and check Streamlit version

Source: https://docs.streamlit.io/knowledge-base/using-streamlit/sanity-checks

This command sequence upgrades Streamlit to the latest version and then displays the currently installed version. This is a primary step to fix issues resolved in newer releases. Requires pip and Streamlit to be installed.

```bash
pip install --upgrade streamlit
streamlit version
```

--------------------------------

### Streamlit Main App File Setup (Python)

Source: https://docs.streamlit.io/develop/tutorials/multipage/st

Sets up the main `app.py` file for a Streamlit application, acting as a pseudo-login page. It initializes `st.session_state.role` and includes a callback function `set_role` to save user role selections.

```Python
import streamlit as st
from menu import menu

# Initialize st.session_state.role to None
if "role" not in st.session_state:
    st.session_state.role = None

# Retrieve the role from Session State to initialize the widget
st.session_state._role = st.session_state.role

def set_role():
    # Callback function to save the role selection to Session State
    st.session_state.role = st.session_state._role

```

--------------------------------

### Install Python Dependencies for Streamlit LLM App

Source: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/llm-quickstart

Installs the necessary Python libraries, streamlit and langchain-openai, required for building the LLM application. This command is run in the terminal.

```bash
pip install streamlit langchain-openai
```

--------------------------------

### Declare Python Dependencies with requirements.txt

Source: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies

Illustrates how to create a `requirements.txt` file to specify Python package dependencies for a Streamlit app. This file is used by pip to install necessary libraries. It shows examples of direct installation, version pinning, and range-based version constraints.

```text
streamlit
pandas
numpy
```

```text
streamlit==1.24.1
pandas>2.0
numpy<=1.25.1
```

--------------------------------

### Initialize Frontend Development Server for Streamlit Component Template

Source: https://docs.streamlit.io/develop/concepts/custom-components/intro

Sets up the development environment for a Streamlit component using the React template. This involves installing Node.js dependencies and starting the Vite development server. It's a crucial step for building bi-directional Streamlit components.

```bash
# React template
template/my_component/frontend
npm install    # Initialize the project and install npm dependencies
npm run start  # Start the Vite dev server

```

--------------------------------

### Verify Streamlit version within a script

Source: https://docs.streamlit.io/knowledge-base/using-streamlit/sanity-checks

This Python script snippet prints the installed Streamlit version. It helps confirm that the Python environment is using the expected Streamlit installation, especially when multiple environments might be present. Requires Streamlit to be installed in the Python environment.

```python
import streamlit as st
st.write(st.__version__)
```

--------------------------------

### Render User-Input Markdown with st.markdown and st.text_area in Python

Source: https://docs.streamlit.io/develop/api-reference/text/st

This example demonstrates how to capture user input using st.text_area and then render that input as markdown using st.markdown. It also displays the generated code for clarity.

```python
import streamlit as st

md = st.text_area('Type in your markdown string (without outer quotes)',
                  "Happy Streamlit-ing! :balloon:")

st.code(f"""
import streamlit as st

st.markdown('''{md}''')
""")

st.markdown(md)
```

--------------------------------

### Streamlit command not recognized (Windows PowerShell)

Source: https://docs.streamlit.io/knowledge-base/using-streamlit/sanity-checks

This example illustrates the error message in Windows PowerShell when the 'streamlit' command is not found. Similar to the CMD error, this usually points to an issue with the system's PATH variable not including the Python installation directory.

```powershell
PS C:\Users\streamlit> streamlit hello
streamlit : The term 'streamlit' is not recognized as the name of a cmdlet, function, script file, or operable program. Check the spelling of the name, or if a path was included, verify that
the path is correct and try again.
At line:1 char:1
+ streamlit hello
+ ~~~~~~~~~ 
    + CategoryInfo          : ObjectNotFound: (streamlit:String) [], CommandNotFoundException
    + FullyQualifiedErrorId : CommandNotFoundException
```

--------------------------------

### Install Streamlit Nightly for Pre-release Features

Source: https://docs.streamlit.io/develop/quick-reference/cheat-sheet

This snippet demonstrates how to uninstall the stable version of Streamlit and install the nightly build to access pre-release and experimental features.

```python
pip uninstall streamlit
pip install streamlit-nightly --upgrade
```

--------------------------------

### Streamlit: Stateful Widgets Across Multiple Pages

Source: https://docs.streamlit.io/develop/api-reference/navigation/st

This example demonstrates how to maintain the state of widgets across different pages in a Streamlit application. By calling widget functions in the entrypoint file and assigning keys, their values can be accessed via `st.session_state` within any page, ensuring a consistent user experience.

```Python
import streamlit as st

def page1():
    st.write(st.session_state.foo)

def page2():
    st.write(st.session_state.bar)

# Widgets shared by all the pages
st.sidebar.selectbox("Foo", ["A", "B", "C"], key="foo")
st.sidebar.checkbox("Bar", key="bar")

pg = st.navigation([page1, page2])
pg.run()
```

--------------------------------

### Organize Streamlit App with Tabs

Source: https://docs.streamlit.io/develop/tutorials/elements/dataframe-row-selections

This example demonstrates how to use Streamlit's `st.tabs` to separate different sections of an application, such as member selection and data comparison. It groups related UI elements and logic within each tab.

```Python
select, compare = st.tabs(["Select members", "Compare selected"])

with select:
    st.header("All members")
    df = get_profile_dataset()
    event = st.dataframe(
        df,
        column_config=column_configuration,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
    )
    st.header("Selected members")
    people = event.selection.rows
    filtered_df = df.iloc[people]
    st.dataframe(
        filtered_df,
        column_config=column_configuration,
        use_container_width=True,
    )

with compare:
    activity_df = {}
    for person in people:
        activity_df[df.iloc[person]["name"]] = df.iloc[person]["activity"]
    activity_df = pd.DataFrame(activity_df)

    daily_activity_df = {}
    for person in people:
        daily_activity_df[df.iloc[person]["name"]] = df.iloc[person]["daily_activity"]
    daily_activity_df = pd.DataFrame(daily_activity_df)

    if len(people) > 0:
        st.header("Daily activity comparison")
        st.bar_chart(daily_activity_df)
        st.header("Yearly activity comparison")
        st.line_chart(activity_df)
    else:
        st.markdown("No members selected.")
```

--------------------------------

### Streamlit Application Example

Source: https://docs.streamlit.io/deploy/tutorials/docker

A basic Streamlit application demonstrating interactive widgets like sliders and displaying a chart using Altair and Pandas. It includes an echo feature to show the code generating the output.

```python
from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st

"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""

with st.echo(code_location='below'):
   total_points = st.slider("Number of points in spiral", 1, 5000, 2000)
   num_turns = st.slider("Number of turns in spiral", 1, 100, 9)

   Point = namedtuple('Point', 'x y')
   data = []

   points_per_turn = total_points / num_turns

   for curr_point_num in range(total_points):
      curr_turn, i = divmod(curr_point_num, points_per_turn)
      angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
      radius = curr_point_num / total_points
      x = radius * math.cos(angle)
      y = radius * math.sin(angle)
      data.append(Point(x, y))

   st.altair_chart(alt.Chart(pd.DataFrame(data), height=500, width=500)
      .mark_circle(color='#0068c9', opacity=0.5)
      .encode(x='x:Q', y='y:Q'))
```

--------------------------------

### Execute Streamlit Tests with Pytest (Terminal)

Source: https://docs.streamlit.io/develop/concepts/app-testing/get-started

Command to navigate to the project's root directory and execute Streamlit tests using pytest. This assumes the test files are located within a 'tests' subdirectory.

```bash
cd myproject
pytest tests/
```

--------------------------------

### Streamlit App Introduction

Source: https://docs.streamlit.io/get-started/tutorials/create-a-multipage-app

The introduction function displays a welcome message, instructions, and links to Streamlit resources. It uses st.write and st.markdown for text formatting and st.sidebar.success for sidebar notifications. No external dependencies beyond Streamlit itself are required.

```python
import streamlit as st

def intro():
    import streamlit as st

    st.write("# Welcome to Streamlit! 👋")
    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        Streamlit is an open-source app framework built specifically for
        Machine Learning and Data Science projects.

        **👈 Select a demo from the dropdown on the left** to see some examples
        of what Streamlit can do!

        ### Want to learn more?

        - Check out [streamlit.io](https://streamlit.io)
        - Jump into our [documentation](https://docs.streamlit.io)
        - Ask a question in our [community
          forums](https://discuss.streamlit.io)

        ### See more complex demos

        - Use a neural net to [analyze the Udacity Self-driving Car Image
          Dataset](https://github.com/streamlit/demo-self-driving)
        - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
    """
    )
```

--------------------------------

### Create Streamlit Entrypoint File (Hello.py)

Source: https://docs.streamlit.io/get-started/tutorials/create-a-multipage-app

This Python script serves as the main entry point for a Streamlit application. It configures the page title and icon, displays a welcome message, and provides introductory text with links to further resources. It uses the `streamlit` library for all functionalities.

```python
import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Streamlit! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    **👈 Select a demo from the sidebar** to see some examples
    of what Streamlit can do!
    ### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
    ### See more complex demos
    - Use a neural net to [analyze the Udacity Self-driving Car Image
        Dataset](https://github.com/streamlit/demo-self-driving)
    - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
"""
)

```

--------------------------------

### Install Streamlit Component using pip

Source: https://docs.streamlit.io/develop/concepts/custom-components

This command installs the 'streamlit-aggrid' component, a popular third-party module for enhancing Streamlit apps with advanced data grid features. Ensure you have pip installed and configured.

```python
pip install streamlit-aggrid
```

--------------------------------

### Dockerfile Base Image Instruction (Docker)

Source: https://docs.streamlit.io/deploy/tutorials/docker

Specifies the base operating system image for the Docker container. This example uses a lightweight Python 3.9 image.

```docker
FROM python:3.9-slim
```

--------------------------------

### Pre-release Features

Source: https://docs.streamlit.io/develop/quick-reference/cheat-sheet

How to install and use the pre-release version of Streamlit for experimental features.

```APIDOC
## Pre-release Features

### Description
Install the nightly build of Streamlit to access pre-release and experimental features.

### Method
Command Line

### Endpoint
N/A

### Parameters
None

### Request Example
```bash
pip uninstall streamlit
pip install streamlit-nightly --upgrade
```

### Response
N/A
```

--------------------------------

### Configuration File Structure

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=configuration&slug=chat

Example of the directory structure for Streamlit configuration files.

```APIDOC
## File Structure

### Description
Configures the default settings for your app using a `config.toml` file.

### Example
```
your-project/
├── .streamlit/
│   └── config.toml
└── your_app.py
```
```

--------------------------------

### Retrieve Streamlit Widgets by Key using AppTest

Source: https://docs.streamlit.io/develop/concepts/app-testing/get-started

Demonstrates how to retrieve specific Streamlit widgets by their assigned keys using the AppTest framework. This allows for targeted testing of individual widgets regardless of their order on the page. The key is passed as an argument to the widget retrieval function.

```Python
import streamlit as st

st.button("Next", key="submit")
st.button("Back", key="cancel")
```

```Python
assert at.button(key="submit").label == "Next"
assert at.button("cancel").label == "Back"
```

--------------------------------

### Display Help for a Variable using st.help

Source: https://docs.streamlit.io/develop/api-reference/utilities/st

This example shows how to use st.help to inspect the type and documentation of a variable, especially useful for functions with unclear return types. It helps in understanding the output of a function or the state of a variable.

```python
import streamlit as st

x = my_poorly_documented_function()
st.help(x)
```

--------------------------------

### Execute pytest Tests in Specific Directory

Source: https://docs.streamlit.io/develop/concepts/app-testing/get-started

Directs pytest to scan and execute tests only within the specified 'tests/' directory. This is useful for organizing tests and controlling which tests are run.

```bash
pytest tests/
```

--------------------------------

### Install V2 Component Library (Terminal)

Source: https://docs.streamlit.io/develop/api-reference/custom-components

Installs the necessary library for developing Streamlit V2 custom components. This command should be run in your project's terminal to add the `@streamlit/component-v2-lib` package.

```bash
npm i @streamlit/component-v2-lib
```

--------------------------------

### Assert Streamlit App State

Source: https://docs.streamlit.io/develop/concepts/app-testing/get-started

Asserts that the value of a Streamlit markdown element matches the expected output after user interactions. This is a core part of testing, verifying that the app behaves as intended.

```python
assert at.markdown[0].value == "Beans counted: 1"
```

--------------------------------

### Dockerfile Working Directory Instruction (Docker)

Source: https://docs.streamlit.io/deploy/tutorials/docker

Sets the default directory within the container where subsequent commands will be executed. This example sets it to '/app'.

```docker
WORKDIR /app
```

--------------------------------

### Custom DuckDB Connection Implementation

Source: https://docs.streamlit.io/develop/concepts/connections/connecting-to-data

Provides a Python example of building a custom Streamlit connection by extending `ExperimentalBaseConnection`. This example specifically implements a connection for DuckDB, demonstrating initialization and connection logic.

```python
from streamlit.connections import ExperimentalBaseConnection
import duckdb

class DuckDBConnection(ExperimentalBaseConnection[duckdb.DuckDBPyConnection]):
    def _connect(self, **kwargs) -> duckdb.DuckDBPyConnection:
        if 'database' in kwargs:
            db = kwargs.pop('database')
        else:
            db = self._secrets['database']
        return duckdb.connect(database=db, **kwargs)
```

--------------------------------

### Install Supabase Python Client Library

Source: https://docs.streamlit.io/develop/tutorials/databases/supabase

This command adds the Supabase Python Client Library to your project's dependencies. It's recommended to pin the version in your `requirements.txt` file for reproducible builds.

```bash
# requirements.txt
supabase==x.x.x
```

--------------------------------

### Implement Multi-Page Apps with Streamlit Pages

Source: https://docs.streamlit.io/develop/api-reference/layout

This example demonstrates an experimental way to create multi-page applications in Streamlit using the `st-pages` library. It shows how to define and display different pages.

```python
from st_pages import Page, show_pages, add_page_title

show_pages([
    Page("streamlit_app.py", "Home", "🏠"),
    Page("other_pages/page2.py", "Page 2", ":books:"),
])
```

--------------------------------

### Control Process Stages with Buttons and Session State in Streamlit

Source: https://docs.streamlit.io/develop/concepts/design/buttons

This example demonstrates how to manage different stages of a user process using Streamlit buttons and session state. It allows users to navigate through a multi-step process, with buttons triggering state changes that control which widgets are displayed. Dependencies include the Streamlit library.

```python
import streamlit as st

if 'stage' not in st.session_state:
    st.session_state.stage = 0

def set_state(i):
    st.session_state.stage = i

if st.session_state.stage == 0:
    st.button('Begin', on_click=set_state, args=[1])

if st.session_state.stage >= 1:
    name = st.text_input('Name', on_change=set_state, args=[2])

if st.session_state.stage >= 2:
    st.write(f'Hello {name}!')
    color = st.selectbox(
        'Pick a Color',
        [None, 'red', 'orange', 'green', 'blue', 'violet'],
        on_change=set_state, args=[3]
    )
    if color is None:
        set_state(2)

if st.session_state.stage >= 3:
    st.write(f':{color}[Thank you!])
    st.button('Start Over', on_click=set_state, args=[0])
```

--------------------------------

### Streamlit Form with Session State and Callbacks

Source: https://docs.streamlit.io/develop/concepts/architecture/session-state

This example demonstrates using Streamlit's forms in conjunction with session state and callbacks. It allows users to input a time and an increment value within a form, and upon submission, updates both the count and the last updated time using a callback function.

```python
import streamlit as st
import datetime

st.title('Counter Example')
if 'count' not in st.session_state:
    st.session_state.count = 0
    st.session_state.last_updated = datetime.time(0,0)

def update_counter():
    st.session_state.count += st.session_state.increment_value
    st.session_state.last_updated = st.session_state.update_time

with st.form(key='my_form'):
    st.time_input(label='Enter the time', value=datetime.datetime.now().time(), key='update_time')
    st.number_input('Enter a value', value=0, step=1, key='increment_value')
    submit = st.form_submit_button(label='Update', on_click=update_counter)

st.write('Current Count = ', st.session_state.count)
st.write('Last Updated = ', st.session_state.last_updated)
```

--------------------------------

### Streamlit Multi-Page Apps Setup

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

An experimental setup for Streamlit Multi-Page Apps using the `st_pages` library. It allows defining and organizing multiple pages within a single Streamlit project.

```python
from st_pages import Page, show_pages, add_page_title

show_pages([
  Page("streamlit_app.py", "Home", "🏠"),
  Page("other_pages/page2.py", "Page 2", ":books:"),
])
```

--------------------------------

### Create Streamlit Dockerfile (Docker)

Source: https://docs.streamlit.io/deploy/tutorials/docker

A sample Dockerfile for building a Streamlit application image. It specifies the base Python image, sets the working directory, installs dependencies, exposes the Streamlit port, and defines health check and entrypoint commands.

```docker
# app/Dockerfile

FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/streamlit/streamlit-example.git .

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

--------------------------------

### Initialize Streamlit App (Python)

Source: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/chat-response-feedback

This Python code snippet initializes a Streamlit application by importing necessary libraries, specifically 'streamlit' for app development and 'time' for potential delays or timing operations. It's a basic setup for creating interactive web applications with Streamlit.

```Python
import streamlit as st
import time
```

--------------------------------

### Run Streamlit Multipage App

Source: https://docs.streamlit.io/get-started/tutorials/create-a-multipage-app

This command executes a Streamlit multipage application. The `streamlit run` command initiates the app, with `Hello.py` serving as the main page. Other scripts in the `pages` directory will be automatically included as navigable pages in the app's sidebar. JavaScript must be enabled for the app to function.

```bash
streamlit run Hello.py
```

--------------------------------

### Set Streamlit Application Entrypoint in Docker

Source: https://docs.streamlit.io/deploy/tutorials/docker

Defines the command that will be executed when the Docker container starts. This command runs the Streamlit application, specifying the script to execute and the server address and port.

```dockerfile
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

--------------------------------

### Downgrade Streamlit to a specific version

Source: https://docs.streamlit.io/knowledge-base/using-streamlit/sanity-checks

This command allows downgrading Streamlit to a specific version, useful for identifying or reverting regressions introduced in newer releases. Replace '1.0.0' with the desired version number. Requires pip and Streamlit to be installed.

```bash
pip install --upgrade streamlit==1.0.0
```

--------------------------------

### Execute pytest Tests in Project Directory

Source: https://docs.streamlit.io/develop/concepts/app-testing/get-started

Executes all tests found within the current project directory using the pytest framework. Pytest automatically discovers test files (named `test_*.py` or `*_test.py`) and test functions (prefixed with `test_`).

```bash
pytest
```

--------------------------------

### Display Help for a Custom Class Instance using st.help

Source: https://docs.streamlit.io/develop/api-reference/utilities/st

This snippet illustrates how to use st.help to get information about an instance of a custom class, including its attributes and methods. This is helpful for understanding the behavior and properties of user-defined objects.

```python
import streamlit as st

class Dog:
  '''A typical dog.'''

  def __init__(self, breed, color):
    self.breed = breed
    self.color = color

  def bark(self):
    return 'Woof!'


fido = Dog("poodle", "white")

st.help(fido)
```

--------------------------------

### Streamlit command not recognized (Windows CMD)

Source: https://docs.streamlit.io/knowledge-base/using-streamlit/sanity-checks

This example shows the error message encountered in the Windows Command Prompt when the 'streamlit' command is not recognized, typically because Python is not added to the system's PATH environment variable. This indicates an environment configuration issue.

```text
C:\Users\streamlit> streamlit hello
'streamlit' is not recognized as an internal or external command,
operable program or batch file.
```

--------------------------------

### Install Streamlit Dependency

Source: https://docs.streamlit.io/develop/tutorials/multipage/dynamic-navigation

This code snippet specifies the minimum required version for the Streamlit library. It ensures that the application runs with compatible features, particularly those related to navigation introduced in version 1.36.0.

```Python
streamlit>=1.36.0
```

--------------------------------

### Inspect Streamlit Selectbox Widget Properties with AppTest

Source: https://docs.streamlit.io/develop/concepts/app-testing/get-started

Shows how to inspect various properties of a Streamlit selectbox widget using AppTest. This includes checking its current value, label, options, help text, placeholder, and disabled status. It highlights how options are cast to strings internally.

```Python
import streamlit as st

st.selectbox("A", [1,2,3], None, help="Pick a number", placeholder="Pick me")
```

```Python
assert at.selectbox[0].value == None
assert at.selectbox[0].label == "A"
assert at.selectbox[0].options == ["1","2","3"]
assert at.selectbox[0].index == None
assert at.selectbox[0].help == "Pick a number"
assert at.selectbox[0].placeholder == "Pick me"
assert at.selectbox[0].disabled == False
```

--------------------------------

### Streamlit App: Bean Counter

Source: https://docs.streamlit.io/develop/concepts/app-testing/get-started

A simple Streamlit application that allows users to add beans to a counter. It uses session state to maintain the bean count and includes elements for title, number input, button, and markdown display.

```python
import streamlit as st

# Initialize st.session_state.beans
st.session_state.beans = st.session_state.get("beans", 0)

st.title("Bean counter :paw_prints:")

addend = st.number_input("Beans to add", 0, 10)
if st.button("Add"):
    st.session_state.beans += addend
st.markdown(f"Beans counted: {st.session_state.beans}")
```

--------------------------------

### Detecting Unserializable Data in Streamlit Session State

Source: https://docs.streamlit.io/develop/concepts/architecture/session-state

This Python code demonstrates how Streamlit raises an exception when attempting to store unserializable data in session state while the `enforceSerializableSessionState` option is enabled. The example uses a lambda function, which is not pickle-serializable by default.

```python
import streamlit as st

def unserializable_data():
		return lambda x: x

#👇 results in an exception when enforceSerializableSessionState is on
st.session_state.unserializable = unserializable_data()
```

--------------------------------

### Add MySQL Dependencies to requirements.txt

Source: https://docs.streamlit.io/develop/tutorials/databases/mysql

Specifies the necessary Python packages, 'mysqlclient' and 'SQLAlchemy', to be installed for the Streamlit application to connect to MySQL. Version pinning is recommended.

```text
# requirements.txt
mysqlclient==x.x.x
SQLAlchemy==x.x.x
```

--------------------------------

### Streamlit App Running Output

Source: https://docs.streamlit.io/deploy/tutorials/docker

Example output shown in the terminal after successfully running the Streamlit Docker container. It indicates that the application is accessible via a URL.

```text
docker run -p 8501:8501 streamlit

  You can now view your Streamlit app in your browser.

  URL: http://0.0.0.0:8501

```

--------------------------------

### Install Python Dependencies from requirements.txt

Source: https://docs.streamlit.io/deploy/tutorials/docker

Installs all Python packages listed in the 'requirements.txt' file within the Docker container using pip. This ensures all necessary libraries for the Streamlit application are available.

```dockerfile
RUN pip3 install -r requirements.txt
```

--------------------------------

### Create Neon Table and Insert Sample Data

Source: https://docs.streamlit.io/develop/tutorials/databases/neon

SQL commands to create a 'home' table with id, name, and pet columns, and insert sample records. This is used to populate the Neon database for the Streamlit app.

```sql
CREATE TABLE home (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    pet VARCHAR(100)
);

INSERT INTO home (name, pet)
VALUES
    ('Mary', 'dog'),
    ('John', 'cat'),
    ('Robert', 'bird');
```

--------------------------------

### Streamlit Session State for Widget State Association

Source: https://docs.streamlit.io/develop/concepts/architecture/session-state

This Streamlit code illustrates how to associate widget states with session state variables. It shows how to set and get the value of a slider widget using `st.session_state.celsius`, simplifying state management across reruns.

```python
import streamlit as st

if "celsius" not in st.session_state:
    # set the initial default value of the slider widget
    st.session_state.celsius = 50.0

st.slider(
    "Temperature in Celsius",
    min_value=-100.0,
    max_value=100.0,
    key="celsius"
)

# This will get the value of the slider widget
st.write(st.session_state.celsius)
```

--------------------------------

### Custom Connection - Base Class

Source: https://docs.streamlit.io/develop/api-reference/connections

Guide on how to build your own custom connection using the `BaseConnection` class.

```APIDOC
## POST /websites/streamlit_io/connections/custom

### Description
Create a custom data source or API connection by inheriting from `st.connection.BaseConnection`.

### Method
POST

### Endpoint
`/websites/streamlit_io/connections/custom`

### Parameters
#### Request Body
- **connection_class** (class) - Required - Your custom connection class inheriting from `BaseConnection`.

### Request Example
```python
from streamlit.connections import BaseConnection

class MyConnection(BaseConnection[myconn.MyConnection]):
    def _connect(self, **kwargs) -> MyConnection:
        return myconn.connect(**self._secrets, **kwargs)
    def query(self, query):
        return self._instance.query(query)

# Instantiate your custom connection
my_conn = st.connection('my_custom_db', type=MyConnection)
```

### Response
#### Success Response (200)
Indicates that the custom connection class has been registered and can be instantiated.

#### Response Example
```json
{
  "message": "Custom connection class registered successfully."
}
```
```

--------------------------------

### Install Streamlit Nightly Release

Source: https://docs.streamlit.io/develop/quick-reference/prerelease

This snippet shows how to install the 'streamlit-nightly' package, which provides the latest Streamlit features and bug fixes. It includes commands to uninstall the stable 'streamlit' package first to avoid conflicts. This is recommended for users who want to test the newest code and help find bugs early.

```bash
pip uninstall streamlit
pip install streamlit-nightly --upgrade
```

--------------------------------

### Activate Virtual Environment

Source: https://docs.streamlit.io/get-started/installation/command-line

Activate the created virtual environment. The command varies by operating system. Once activated, your terminal prompt will show the environment name.

```bash
# Windows command prompt
.venv\Scripts\activate.bat

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS and Linux
source .venv/bin/activate
```

--------------------------------

### Create PostgreSQL Table

Source: https://docs.streamlit.io/develop/tutorials/databases/postgresql

SQL commands to create a table named 'mytable' with 'name' and 'pet' columns and insert sample data. This is a prerequisite for the Streamlit application example.

```sql
CREATE TABLE mytable (
    name            varchar(80),
    pet             varchar(80)
);

INSERT INTO mytable VALUES ('Mary', 'dog'), ('John', 'cat'), ('Robert', 'bird');
```

--------------------------------

### Streamlit Test: Increment and Add Beans

Source: https://docs.streamlit.io/develop/concepts/app-testing/get-started

A pytest test function for a Streamlit app. It simulates a user incrementing a number input, clicking an 'Add' button, and then asserts that the displayed bean count is updated correctly. Requires Streamlit's AppTest utility.

```python
from streamlit.testing.v1 import AppTest

def test_increment_and_add():
    """A user increments the number input, then clicks Add"""
    at = AppTest.from_file("app.py").run()
    at.number_input[0].increment().run()
    at.button[0].click().run()
    assert at.markdown[0].value == "Beans counted: 1"
```

--------------------------------

### Initialize Streamlit App

Source: https://docs.streamlit.io/develop/tutorials/configuration-and-theming/external-fonts

This snippet shows how to create a basic Streamlit app file and run it from the terminal. It requires the Streamlit library to be installed. The initial app will be blank until content is added.

```shell
streamlit run streamlit_app.py
```

```python
import streamlit as st
```

--------------------------------

### Streamlit Run Command

Source: https://docs.streamlit.io/develop/api-reference/cli/run

The `streamlit run` command is used to start your Streamlit application. It can take an entrypoint file or directory, configuration options, and script arguments.

```APIDOC
## POST /run

### Description
Starts a Streamlit application from a specified entrypoint file or directory. Supports configuration options and script arguments.

### Method
POST

### Endpoint
/run

### Parameters
#### Query Parameters
- **entrypoint_file_or_directory** (string) - Optional - The path to your Streamlit app's entrypoint file or directory. If not provided, Streamlit defaults to `streamlit_app.py` in the current working directory.
- **config_options** (string) - Optional - Configuration options in the format `--<section>.<option>=<value>`.
- **script_args** (string) - Optional - Arguments to be passed directly to your script.

### Request Example
```json
{
  "entrypoint_file_or_directory": "your_app.py",
  "config_options": [
    "--theme.primaryColor=blue",
    "--client.showErrorDetails=False"
  ],
  "script_args": [
    "argument1",
    "argument2"
  ]
}
```

### Response
#### Success Response (200)
- **message** (string) - A confirmation message indicating the app has started.

#### Response Example
```json
{
  "message": "Streamlit app started successfully."
}
```

#### Error Response (400)
- **error** (string) - A message describing the error (e.g., invalid path, configuration error).

#### Error Response Example
```json
{
  "error": "Invalid entrypoint file path provided."
}
```
```

--------------------------------

### Serve Static Files in Streamlit

Source: https://docs.streamlit.io/get-started/fundamentals/additional-features

This example shows how to serve static files, such as images, directly from a Streamlit application. By placing files in a `static` directory within your project and configuring Streamlit correctly, you can provide direct URLs to these assets. This is useful when you need direct access to files that Streamlit commands do not handle automatically.

```Terminal
your-project/
├── static/
│   └── my_hosted-image.png
└── streamlit_app.py
```

--------------------------------

### Run Streamlit App (Basic)

Source: https://docs.streamlit.io/develop/api-reference/cli/run

Starts a Streamlit app using the default `streamlit_app.py` in the current directory. This is the simplest way to launch an app.

```bash
streamlit run
```

--------------------------------

### Example Font CSS URL (Text)

Source: https://docs.streamlit.io/develop/tutorials/configuration-and-theming/external-fonts

An example of a CSS URL obtained from Google Fonts for the 'Nunito' font, used in the Streamlit configuration.

```text
https://fonts.googleapis.com/css2?family=Nunito:ital,wght@0,200..1000;1,200..1000
```

--------------------------------

### Streamlit Fragment: Basic Balloon Release Example

Source: https://docs.streamlit.io/develop/api-reference/execution-flow/st

Demonstrates basic usage of `@st.fragment`. It simulates a slow process outside the fragment ('Inflating balloons') and a quick process inside the fragment ('Releasing balloons'). The fragment rerun is triggered by a button click within the fragment.

```python
import streamlit as st
import time

@st.fragment
def release_the_balloons():
    st.button("Release the balloons", help="Fragment rerun")
    st.balloons()

with st.spinner("Inflating balloons..."):
    time.sleep(5)
release_the_balloons()
st.button("Inflate more balloons", help="Full rerun")
```

--------------------------------

### Install Python Packages for Streamlit-Snowflake Connection

Source: https://docs.streamlit.io/develop/tutorials/databases/snowflake

Installs necessary Python packages for integrating Streamlit with Snowflake. Ensure these versions are compatible with your Python environment.

```text
streamlit>=1.28
snowflake-snowpark-python>=0.9.0
snowflake-connector-python>=2.8.0
```

--------------------------------

### Streamlit App: Control Live Data Streaming with Fragments

Source: https://docs.streamlit.io/develop/tutorials/execution-flow/start-and-stop-fragment-auto-reruns

This Python code demonstrates how to create a Streamlit app that streams live data to a line chart. It uses `st.session_state` to manage data and streaming status, and `st.fragment` with a dynamic `run_every` parameter to control the update frequency. Buttons in the sidebar allow users to start and stop the streaming.

```Python
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def get_recent_data(last_timestamp):
    """Generate and return data from last timestamp to now, at most 60 seconds."""
    now = datetime.now()
    if now - last_timestamp > timedelta(seconds=60):
        last_timestamp = now - timedelta(seconds=60)
    sample_time = timedelta(seconds=0.5)  # time between data points
    next_timestamp = last_timestamp + sample_time
    timestamps = np.arange(next_timestamp, now, sample_time)
    sample_values = np.random.randn(len(timestamps), 2) # Two data series

    data = pd.DataFrame(sample_values, index=timestamps, columns=["A", "B"])
    return data


if "data" not in st.session_state:
    st.session_state.data = get_recent_data(datetime.now() - timedelta(seconds=60))

if "stream" not in st.session_state:
    st.session_state.stream = False


def toggle_streaming():
    st.session_state.stream = not st.session_state.stream

st.title("Data feed")
st.sidebar.slider(
    "Check for updates every: (seconds)", 0.5, 5.0, value=1.0, key="run_every"
)
st.sidebar.button(
    "Start streaming", disabled=st.session_state.stream, on_click=toggle_streaming
)
st.sidebar.button(
    "Stop streaming", disabled=not st.session_state.stream, on_click=toggle_streaming
)

if st.session_state.stream is True:
    run_every = st.session_state.run_every
else:
    run_every = None


@st.fragment(run_every=run_every)
def show_latest_data():
    last_timestamp = st.session_state.data.index[-1]
    st.session_state.data = pd.concat(
        [st.session_state.data, get_recent_data(last_timestamp)]
    )
    st.session_state.data = st.session_state.data[-100:] # Keep only the last 100 data points
    st.line_chart(st.session_state.data)


show_latest_data()

```

--------------------------------

### Getting Help with st.help

Source: https://docs.streamlit.io/develop/api-reference_slug=private-gsheet

Display an object's docstring in a nicely formatted way using `st.help`.

```APIDOC
## POST /st.help

### Description
Display object’s doc string, nicely formatted.

### Method
POST

### Endpoint
/st.help

### Parameters
#### Request Body
- **object_name** (string) - Required - The name of the object to get help for (e.g., 'st.write', 'pd.DataFrame').

### Request Example
```json
{
  "object_name": "st.write"
}
```
```json
{
  "object_name": "pd.DataFrame"
}
```

### Response
#### Success Response (200)
- **docstring** (string) - The formatted docstring of the object.

#### Response Example
```json
{
  "docstring": "Help text for st.write..."
}
```
```

--------------------------------

### Create Python Virtual Environment with venv

Source: https://docs.streamlit.io/get-started/installation/command-line

Create a virtual environment named '.venv' in your project directory using Python's built-in venv module. This isolates project dependencies.

```bash
python -m venv .venv
```

--------------------------------

### Expected Output with st.cache_data for Concurrency Example in Python

Source: https://docs.streamlit.io/develop/concepts/architecture/caching

This code demonstrates the expected output when using st.cache_data for the list mutation example. st.cache_data creates a copy for each user, preventing concurrent modifications from affecting other sessions and ensuring consistent results.

```python
import streamlit as st

@st.cache_data  # Use st.cache_data to get a copy
def create_list():
    l = [1, 2, 3]
    return l

l = create_list()
first_list_value = l[0]
l[0] = first_list_value + 1

st.write("l[0] is:", l[0])
```

--------------------------------

### Create Snowflake Database and Table using SQL

Source: https://docs.streamlit.io/develop/tutorials/databases/snowflake

SQL statements to create a new database named 'PETS', a table 'MYTABLE' with 'NAME' and 'PET' columns, and insert sample data. Includes a query to select all data from the table.

```sql
CREATE DATABASE PETS;

CREATE TABLE MYTABLE (NAME varchar(80), PET varchar(80));

INSERT INTO MYTABLE
VALUES ('Mary', 'dog'), ('John', 'cat'), ('Robert', 'bird');

SELECT * FROM MYTABLE;
```

--------------------------------

### Open Streamlit Documentation

Source: https://docs.streamlit.io/develop/api-reference/cli/docs

Opens the Streamlit documentation in your default web browser. This command is useful for quickly accessing help and guides related to Streamlit.

```bash
streamlit docs
```

--------------------------------

### Activate Anaconda Environment

Source: https://docs.streamlit.io/get-started/installation/anaconda-distribution

Activates a specific Anaconda environment named 'streamlitenv'. This command must be run before installing or running Streamlit within that environment.

```bash
conda activate streamlitenv
```

--------------------------------

### Project File Structure for Streamlit Deployment

Source: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/llm-quickstart

Illustrates the required file structure for deploying the Streamlit application to Streamlit Community Cloud. It includes the main application file and the requirements file.

```text
your-repository/
├── streamlit_app.py
└── requirements.txt
```

--------------------------------

### Interact with Sidebar Block (Python)

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

Demonstrates interaction with the sidebar block in `AppTest`. This example shows how to click a button within the sidebar and run the app to observe the effect.

```python
# at.sidebar returns a Block
at.sidebar.button[0].click().run()
assert not at.exception
```

--------------------------------

### Attempt to Modify Widget State After Rendering in Streamlit (Error Example)

Source: https://docs.streamlit.io/develop/concepts/design/buttons

This example illustrates an error scenario in Streamlit where an attempt is made to modify a widget's state (via `st.session_state`) after that widget has already been rendered on the page. This will cause an error because the state cannot be changed after the widget has been processed. Dependencies include Streamlit.

```python
import streamlit as st

st.text_input('Name', key='name')

# These buttons will error because their nested code changes
# a widget's state after that widget within the script.
if st.button('Clear name'):
    st.session_state.name = ''
if st.button('Streamlit!'):
    st.session_state.name = ('Streamlit')
```

--------------------------------

### Streamlit: Connect to Different Databases Using Environment Variables

Source: https://docs.streamlit.io/develop/concepts/connections/connecting-to-data

This example demonstrates how to configure Streamlit's `st.connection` to dynamically connect to different database instances (e.g., local vs. staging) based on an environment variable. The secrets file defines connection URLs for different named connections, and the Python code uses the `env:` prefix to specify which connection to use.

```toml
# ~/.streamlit/secrets.toml

[connections.local]
url = "mysql://me:****@localhost:3306/local_db"

[connections.staging]
url = "mysql://jdoe:******@staging.acmecorp.com:3306/staging_db"
```

```python
# streamlit_app.py
import streamlit as st

conn = st.connection("env:DB_CONN", "sql")
df = conn.query("select * from mytable")
# ...
```

```shell
# connect to local
DB_CONN=local streamlit run streamlit_app.py

# connect to staging
DB_CONN=staging streamlit run streamlit_app.py
```

--------------------------------

### Streamlit App Entrypoint Script

Source: https://docs.streamlit.io/deploy/tutorials/kubernetes

A bash script that serves as the entrypoint for a Docker container running a Streamlit application. It handles stopping the running process gracefully and activates a virtual environment before starting the Streamlit app.

```bash
#!/bin/bash

APP_PID=
stopRunningProcess() {
    # Based on https://linuxconfig.org/how-to-propagate-a-signal-to-child-processes-from-a-bash-script
    if test ! "${APP_PID}" = '' && ps -p ${APP_PID} > /dev/null ; then
       > /proc/1/fd/1 echo "Stopping ${COMMAND_PATH} which is running with process ID ${APP_PID}"

       kill -TERM ${APP_PID}
       > /proc/1/fd/1 echo "Waiting for ${COMMAND_PATH} to process SIGTERM signal"

        wait ${APP_PID}
        > /proc/1/fd/1 echo "All processes have stopped running"
    else
        > /proc/1/fd/1 echo "${COMMAND_PATH} was not started when the signal was sent or it has already been stopped"
    fi
}

trap stopRunningProcess EXIT TERM

source ${VIRTUAL_ENV}/bin/activate

streamlit run ${HOME}/app/streamlit_app.py &
APP_ID=${!}

wait ${APP_ID}

```

--------------------------------

### Basic Streamlit App Example

Source: https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/trust-and-security_slug=deploy&slug=streamlit-community-cloud&slug=get-started

A simple Streamlit application demonstrating the use of a slider widget and displaying text. This code requires the Streamlit library to run.

```python
import streamlit as st
number = st.slider("Pick a number: ", min_value=1, max_value=10)
st.text("Your number is " + str(number))
```

--------------------------------

### Implement Basic Streamlit Login Flow (Python)

Source: https://docs.streamlit.io/develop/concepts/connections/authentication

This Python code demonstrates a basic login flow for a Streamlit application using OIDC. It checks if the user is logged in and displays a login button if not. The `st.stop()` function halts script execution until login is complete. This example uses the default OIDC provider.

```python
import streamlit as st

if not st.user.is_logged_in:
    if st.button("Log in with Google"):
        st.login()
    st.stop()

if st.button("Log out"):
    st.logout()
st.markdown(f"Welcome! {st.user.name}")

```

--------------------------------

### Example Code Font CSS URL (Text)

Source: https://docs.streamlit.io/develop/tutorials/configuration-and-theming/external-fonts

An example of a CSS URL obtained from Google Fonts for the 'Space Mono' font, used for code elements in the Streamlit configuration.

```text
https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400;1,700&display=swap
```

--------------------------------

### Show Temporary Message with Streamlit Button

Source: https://docs.streamlit.io/develop/concepts/design/buttons

This example shows how to use st.button to display a temporary message based on user input. The message appears only when the button is clicked and disappears on the next script rerun, illustrating the stateless nature of st.button.

```python
import streamlit as st

animal_shelter = ['cat', 'dog', 'rabbit', 'bird']

animal = st.text_input('Type an animal')

if st.button('Check availability'):
    have_it = animal.lower() in animal_shelter
    'We have that animal!' if have_it else 'We don't have that animal.'

```

--------------------------------

### List Docker Images

Source: https://docs.streamlit.io/deploy/tutorials/docker

Lists all Docker images available on the system. This command is used to verify that the 'streamlit' image was successfully built and is available for use.

```docker
docker images
```

--------------------------------

### Build Python Wheel for Streamlit Component

Source: https://docs.streamlit.io/develop/concepts/custom-components/publish

This command builds a source distribution (sdist) and a wheel (.whl) file for a Python package, which in this case is a Streamlit Component. These files are the distributable artifacts used for uploading to PyPI. Ensure setuptools and wheel are installed.

```bash
# Run this from your component's top-level directory; that is,
# the directory that contains `setup.py`
python setup.py sdist bdist_wheel
```

--------------------------------

### Dynamically Add Widgets with Unique Keys

Source: https://docs.streamlit.io/develop/concepts/design/buttons

When adding multiple instances of the same widget dynamically, it's crucial to ensure each widget has a unique key to prevent `DuplicateWidgetID` errors. This example defines a function `display_input_row` that accepts an index, which is then incorporated into the keys of the text input widgets it creates. This allows the function to be called multiple times per script rerun without key conflicts.

```python
import streamlit as st

def display_input_row(index):
    left, middle, right = st.columns(3)
    left.text_input('First', key=f'first_{index}')
    middle.text_input('Middle', key=f'middle_{index}')
    right.text_input('Last', key=f'last_{index}')

if 'rows' not in st.session_state:
    st.session_state['rows'] = 0

def increase_rows():
    st.session_state['rows'] += 1

st.button('Add person', on_click=increase_rows)

for i in range(st.session_state['rows']):
    display_input_row(i)

# Show the results
st.subheader('People')
for i in range(st.session_state['rows']):
    st.write(
        f'Person {i+1}:',
        st.session_state[f'first_{i}'],
        st.session_state[f'middle_{i}'],
        st.session_state[f'last_{i}']
    )
```

--------------------------------

### Create Stateful Button Behavior with Streamlit Session State

Source: https://docs.streamlit.io/develop/concepts/design/buttons

This example demonstrates how to make a Streamlit button retain its 'clicked' state. It uses st.session_state and a callback function to set a session state variable to True when the button is clicked, allowing subsequent elements to remain visible.

```python
import streamlit as st

if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True

st.button('Click me', on_click=click_button)

if st.session_state.clicked:
    # The message and nested widget will remain on the page
    st.write('Button clicked!')
    st.slider('Select a value')

```

--------------------------------

### Streamlit Cache Mutation Example (Python)

Source: https://docs.streamlit.io/develop/concepts/architecture/caching

Illustrates a scenario where the return value of a Streamlit cached function (`st.cache_data`) is mutated after being called. This example highlights potential issues when modifying cached objects directly.

```python
@st.cache_data
def create_list():
    l = [1, 2, 3]

l = create_list()  # 👈 Call the function
l[0] = 2  # 👈 Mutate its return value

```

--------------------------------

### Defining a Pydantic Model for Streamlit Caching Example

Source: https://docs.streamlit.io/develop/concepts/architecture/caching

Defines a simple Pydantic `BaseModel` named `Person` with a single string attribute `name`. This model is used in Streamlit caching examples to illustrate hashing issues and solutions.

```python
import streamlit as st
from pydantic import BaseModel

class Person(BaseModel):
    name: str
```

--------------------------------

### Run Streamlit App (Specific File)

Source: https://docs.streamlit.io/develop/api-reference/cli/run

Starts a Streamlit app by specifying a particular Python file as the entry point. This is useful when your app file is not named `streamlit_app.py` or is in a different location.

```bash
streamlit run your_app.py
```

--------------------------------

### AppTest - Interacting with Elements

Source: https://docs.streamlit.io/develop/api-reference_slug=advanced-features&slug=prerelease

Provides examples of how to interact with various Streamlit elements within the `AppTest` framework.

```APIDOC
## POST /api/testing/apptest/{app_test_id}/secrets/set

### Description
Sets a secret value for the simulated app.

### Method
POST

### Endpoint
`/api/testing/apptest/{app_test_id}/secrets/set`

### Parameters
#### Path Parameters
- **app_test_id** (string) - Required - The identifier of the AppTest instance.

#### Request Body
- **key** (string) - Required - The name of the secret.
- **value** (string) - Required - The value of the secret.

### Request Example
```json
{
  "key": "WORD",
  "value": "Foobar"
}
```

### Response
#### Success Response (200)
Indicates that the secret was set successfully.

## POST /api/testing/apptest/{app_test_id}/sidebar/button/{index}/click

### Description
Simulates a click event on a button within the sidebar.

### Method
POST

### Endpoint
`/api/testing/apptest/{app_test_id}/sidebar/button/{index}/click`

### Parameters
#### Path Parameters
- **app_test_id** (string) - Required - The identifier of the AppTest instance.
- **index** (integer) - Required - The index of the button in the sidebar.

### Request Example
`POST /api/testing/apptest/test_12345/sidebar/button/0/click`

### Response
#### Success Response (200)
Indicates that the button click was simulated.

## POST /api/testing/apptest/{app_test_id}/title/{index}/value

### Description
Retrieves the value of a title element.

### Method
GET

### Endpoint
`/api/testing/apptest/{app_test_id}/title/{index}/value`

### Parameters
#### Path Parameters
- **app_test_id** (string) - Required - The identifier of the AppTest instance.
- **index** (integer) - Required - The index of the title element.

### Response
#### Success Response (200)
- **value** (string) - The text content of the title element.

## POST /api/testing/apptest/{app_test_id}/button/{index}/click

### Description
Simulates a click event on a button.

### Method
POST

### Endpoint
`/api/testing/apptest/{app_test_id}/button/{index}/click`

### Parameters
#### Path Parameters
- **app_test_id** (string) - Required - The identifier of the AppTest instance.
- **index** (integer) - Required - The index of the button.

### Request Example
`POST /api/testing/apptest/test_12345/button/0/click`

### Response
#### Success Response (200)
Indicates that the button click was simulated.

## POST /api/testing/apptest/{app_test_id}/chat_input/{index}/set_value

### Description
Sets the value of a chat input element.

### Method
POST

### Endpoint
`/api/testing/apptest/{app_test_id}/chat_input/{index}/set_value`

### Parameters
#### Path Parameters
- **app_test_id** (string) - Required - The identifier of the AppTest instance.
- **index** (integer) - Required - The index of the chat input element.

#### Request Body
- **value** (string) - Required - The value to set for the chat input.

### Request Example
```json
{
  "value": "What is Streamlit?"
}
```

### Response
#### Success Response (200)
Indicates that the chat input value was set.

## POST /api/testing/apptest/{app_test_id}/checkbox/{index}/check

### Description
Checks a checkbox element.

### Method
POST

### Endpoint
`/api/testing/apptest/{app_test_id}/checkbox/{index}/check`

### Parameters
#### Path Parameters
- **app_test_id** (string) - Required - The identifier of the AppTest instance.
- **index** (integer) - Required - The index of the checkbox.

### Request Example
`POST /api/testing/apptest/test_12345/checkbox/0/check`

### Response
#### Success Response (200)
Indicates that the checkbox was checked.

## POST /api/testing/apptest/{app_test_id}/color_picker/{index}/pick

### Description
Sets the color of a color picker element.

### Method
POST

### Endpoint
`/api/testing/apptest/{app_test_id}/color_picker/{index}/pick`

### Parameters
#### Path Parameters
- **app_test_id** (string) - Required - The identifier of the AppTest instance.
- **index** (integer) - Required - The index of the color picker.

#### Request Body
- **color** (string) - Required - The color value (e.g., hex code).

### Request Example
```json
{
  "color": "#FF4B4B"
}
```

### Response
#### Success Response (200)
Indicates that the color picker value was set.

## POST /api/testing/apptest/{app_test_id}/date_input/{index}/set_value

### Description
Sets the value of a date input element.

### Method
POST

### Endpoint
`/api/testing/apptest/{app_test_id}/date_input/{index}/set_value`

### Parameters
#### Path Parameters
- **app_test_id** (string) - Required - The identifier of the AppTest instance.
- **index** (integer) - Required - The index of the date input element.

#### Request Body
- **date** (string) - Required - The date value (YYYY-MM-DD format).

### Request Example
```json
{
  "date": "2023-10-26"
}
```

### Response
#### Success Response (200)
Indicates that the date input value was set.

## POST /api/testing/apptest/{app_test_id}/multiselect/{index}/select

### Description
Selects an option in a multiselect element.

### Method
POST

### Endpoint
`/api/testing/apptest/{app_test_id}/multiselect/{index}/select`

### Parameters
#### Path Parameters
- **app_test_id** (string) - Required - The identifier of the AppTest instance.
- **index** (integer) - Required - The index of the multiselect element.

#### Request Body
- **option** (string) - Required - The option to select.

### Request Example
```json
{
  "option": "New York"
}
```

### Response
#### Success Response (200)
Indicates that the option was selected.

## POST /api/testing/apptest/{app_test_id}/number_input/{index}/increment

### Description
Increments the value of a number input element.

### Method
POST

### Endpoint
`/api/testing/apptest/{app_test_id}/number_input/{index}/increment`

### Parameters
#### Path Parameters
- **app_test_id** (string) - Required - The identifier of the AppTest instance.
- **index** (integer) - Required - The index of the number input element.

### Request Example
`POST /api/testing/apptest/test_12345/number_input/0/increment`

### Response
#### Success Response (200)
Indicates that the number input was incremented.

## POST /api/testing/apptest/{app_test_id}/radio/{index}/set_value

### Description
Sets the selected value of a radio button group.

### Method
POST

### Endpoint
`/api/testing/apptest/{app_test_id}/radio/{index}/set_value`

### Parameters
#### Path Parameters
- **app_test_id** (string) - Required - The identifier of the AppTest instance.
- **index** (integer) - Required - The index of the radio button group.

#### Request Body
- **value** (string) - Required - The value to set.

### Request Example
```json
{
  "value": "New York"
}
```

### Response
#### Success Response (200)
Indicates that the radio button value was set.

## POST /api/testing/apptest/{app_test_id}/select_slider/{index}/set_range

### Description
Sets the selected range for a select slider element.

### Method
POST

### Endpoint
`/api/testing/apptest/{app_test_id}/select_slider/{index}/set_range`

### Parameters
#### Path Parameters
- **app_test_id** (string) - Required - The identifier of the AppTest instance.
- **index** (integer) - Required - The index of the select slider element.

#### Request Body
- **start** (string) - Required - The start of the range.
- **end** (string) - Required - The end of the range.

### Request Example
```json
{
  "start": "A",
  "end": "C"
}
```

### Response
#### Success Response (200)
Indicates that the select slider range was set.

## POST /api/testing/apptest/{app_test_id}/selectbox/{index}/select

### Description
Selects an option in a selectbox element.

### Method
POST

### Endpoint
`/api/testing/apptest/{app_test_id}/selectbox/{index}/select`

### Parameters
#### Path Parameters
- **app_test_id** (string) - Required - The identifier of the AppTest instance.
- **index** (integer) - Required - The index of the selectbox element.

#### Request Body
- **option** (string) - Required - The option to select.

### Request Example
```json
{
  "option": "New York"
}
```

### Response
#### Success Response (200)
Indicates that the option was selected.

## POST /api/testing/apptest/{app_test_id}/slider/{index}/set_range

### Description
Sets the selected range for a slider element.

### Method
POST

### Endpoint
`/api/testing/apptest/{app_test_id}/slider/{index}/set_range`

### Parameters
#### Path Parameters
- **app_test_id** (string) - Required - The identifier of the AppTest instance.
- **index** (integer) - Required - The index of the slider element.

#### Request Body
- **start** (number) - Required - The start of the range.
- **end** (number) - Required - The end of the range.

### Request Example
```json
{
  "start": 2,
  "end": 5
}
```

### Response
#### Success Response (200)
Indicates that the slider range was set.

## POST /api/testing/apptest/{app_test_id}/text_area/{index}/input

### Description
Inputs text into a text area element.

### Method
POST

### Endpoint
`/api/testing/apptest/{app_test_id}/text_area/{index}/input`

### Parameters
#### Path Parameters
- **app_test_id** (string) - Required - The identifier of the AppTest instance.
- **index** (integer) - Required - The index of the text area element.

#### Request Body
- **text** (string) - Required - The text to input.

### Request Example
```json
{
  "text": "Streamlit is awesome!"
}
```

### Response
#### Success Response (200)
Indicates that the text was input successfully.

## POST /api/testing/apptest/{app_test_id}/text_input/{index}/input

### Description
Inputs text into a text input element.

### Method
POST

### Endpoint
`/api/testing/apptest/{app_test_id}/text_input/{index}/input`

### Parameters
#### Path Parameters
- **app_test_id** (string) - Required - The identifier of the AppTest instance.
- **index** (integer) - Required - The index of the text input element.

#### Request Body
- **text** (string) - Required - The text to input.

### Request Example
```json
{
  "text": "Streamlit"
}
```

### Response
#### Success Response (200)
Indicates that the text was input successfully.

## POST /api/testing/apptest/{app_test_id}/time_input/{index}/increment

### Description
Increments the time value of a time input element.

### Method
POST

### Endpoint
`/api/testing/apptest/{app_test_id}/time_input/{index}/increment`

### Parameters
#### Path Parameters
- **app_test_id** (string) - Required - The identifier of the AppTest instance.
- **index** (integer) - Required - The index of the time input element.

### Request Example
`POST /api/testing/apptest/test_12345/time_input/0/increment`

### Response
#### Success Response (200)
Indicates that the time input was incremented.

## POST /api/testing/apptest/{app_test_id}/toggle/{index}/set_value

### Description
Sets the value of a toggle element.

### Method
POST

### Endpoint
`/api/testing/apptest/{app_test_id}/toggle/{index}/set_value`

### Parameters
#### Path Parameters
- **app_test_id** (string) - Required - The identifier of the AppTest instance.
- **index** (integer) - Required - The index of the toggle element.

#### Request Body
- **value** (string) - Required - The value to set (e.g., "True" or "False").

### Request Example
```json
{
  "value": "True"
}
```

### Response
#### Success Response (200)
Indicates that the toggle value was set.
```

--------------------------------

### Create Draggable Dashboards with Streamlit Elements

Source: https://docs.streamlit.io/develop/api-reference/layout

This example uses the `streamlit-elements` library to create a draggable and resizable dashboard. It integrates Material-UI components within Streamlit.

```python
from streamlit_elements import elements, mui, html

with elements("new_element"):
  mui.Typography("Hello world")
```

--------------------------------

### Initialize Streamlit Session State Variables

Source: https://docs.streamlit.io/develop/concepts/architecture/session-state

This Python code shows how to initialize variables in Streamlit's Session State. It checks if a key exists and, if not, assigns it an initial value. Both dictionary-like and attribute-based syntaxes are demonstrated.

```python
import streamlit as st

# Check if 'key' already exists in session_state
# If not, then initialize it
if 'key' not in st.session_state:
    st.session_state['key'] = 'value'

# Session State also supports the attribute based syntax
if 'key' not in st.session_state:
    st.session_state.key = 'value'
```

--------------------------------

### Implement Toggle Button with Streamlit Session State

Source: https://docs.streamlit.io/develop/concepts/design/buttons

This example shows how to create a toggle button effect using st.button and st.session_state. A callback function flips a boolean value in session state, which then conditionally displays or hides other Streamlit widgets like st.slider.

```python
import streamlit as st

if 'button' not in st.session_state:
    st.session_state.button = False

def click_button():
    st.session_state.button = not st.session_state.button

st.button('Click me', on_click=click_button)

if st.session_state.button:
    # The message and nested widget will remain on the page
    st.write('Button is on!')
    st.slider('Select a value')
else:
    st.write('Button is off!')

```

--------------------------------

### Generate Session-Specific Random Data with Streamlit Session State

Source: https://docs.streamlit.io/get-started/fundamentals/advanced-concepts

This example shows how to generate and store random data using Streamlit's Session State. A pandas DataFrame with random numbers is created only if it doesn't exist in the session state. This ensures that each user session gets unique random data upon initial load, and this data remains consistent throughout their interaction, even when widgets are updated. It also demonstrates linking session state data to a scatter plot and a color picker widget.

```python
import streamlit as st
import pandas as pd
import numpy as np

if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(np.random.randn(20, 2), columns=["x", "y"])

st.header("Choose a datapoint color")
color = st.color_picker("Color", "#FF0000")
st.divider()
st.scatter_chart(st.session_state.df, x="x", y="y", color=color)
```

--------------------------------

### Streamlit Multi-Provider Login with Callbacks (Python)

Source: https://docs.streamlit.io/develop/concepts/connections/authentication

This Python code provides a streamlined Streamlit login flow for multiple OIDC providers using callbacks. The `on_click` argument of `st.button` is configured with `st.login` and the specific provider name passed via `args`, simplifying the UI logic for selecting an authentication provider.

```python
import streamlit as st

if not st.user.is_logged_in:
    st.button("Log in with Google", on_click=st.login, args=["google"])
    st.button("Log in with Microsoft", on_click=st.login, args=["microsoft"])
    st.stop()

st.button("Log out", on_click=st.logout)
st.markdown(f"Welcome! {st.user.name}")

```

--------------------------------

### Third-Party Components

Source: https://docs.streamlit.io/develop/api-reference_slug=%28&slug=develop&slug=concepts&slug=configuration

Examples of third-party components that can be integrated into Streamlit apps.

```APIDOC
### Third-party components

#### st_tags (by @gagan3012)

**Description**: Add tags to your Streamlit apps.

**Method**: Python (Function Call)

**Endpoint**: N/A (Python function)

### Request Example
```python
st_tags(label='# Enter Keywords:', text='Press enter to add more', value=['Zero', 'One', 'Two'], suggestions=['five', 'six', 'seven', 'eight', 'nine', 'three', 'eleven', 'ten', 'four'], maxtags = 4, key='1')
```

### Response
**Success Response**: Displays an input field for adding tags with suggestions and a maximum tag limit.

#### NLU (by @JohnSnowLabs)

**Description**: Apply text mining on a dataframe.

**Method**: Python (Function Call)

**Endpoint**: N/A (Python function)

### Request Example
```python
nlu.load("sentiment").predict("I love NLU! <3")
```

### Response
**Success Response**: Performs NLU operations (e.g., sentiment analysis) on the input text.

#### mention (from Streamlit Extras by @arnaudmiribel)

**Description**: Creates a mention link with an icon and URL.

**Method**: Python (Function Call)

**Endpoint**: N/A (Python function)

### Request Example
```python
mention(label="An awesome Streamlit App", icon="streamlit", url="https://extras.streamlit.app",)
```

### Response
**Success Response**: Displays a clickable mention with a label, icon, and URL.

#### annotated_text (by @tvst)

**Description**: Display annotated text in Streamlit apps.

**Method**: Python (Function Call)

**Endpoint**: N/A (Python function)

### Request Example
```python
annotated_text("This ", ("is", "verb"), " some ", ("annotated", "adj"), ("text", "noun"), " for those of ", ("you", "pronoun"), " who ", ("like", "verb"), " this sort of ", ("thing", "noun"), ".")
```

### Response
**Success Response**: Displays text with specific words annotated with their part of speech or other labels.

#### st_canvas (by @andfanilo)

**Description**: Provides a sketching canvas using Fabric.js.

**Method**: Python (Function Call)

**Endpoint**: N/A (Python function)

### Request Example
```python
st_canvas(fill_color="rgba(255, 165, 0, 0.3)", stroke_width=stroke_width, stroke_color=stroke_color, background_color=bg_color, background_image=Image.open(bg_image) if bg_image else None, update_streamlit=realtime_update, height=150, drawing_mode=drawing_mode, point_display_radius=point_display_radius if drawing_mode == 'point' else 0, key="canvas",)
```

### Response
**Success Response**: Renders an interactive canvas for drawing with various customization options.
```

--------------------------------

### Handle Expensive Processes with Session State

Source: https://docs.streamlit.io/develop/concepts/design/buttons

This pattern is used to manage computationally intensive or time-consuming processes. By running the process upon a button click and storing its results in `st.session_state`, you can avoid re-executing the process on subsequent reruns, improving performance. This is particularly useful for operations that involve file I/O or database interactions. The example demonstrates saving results based on a parameter to avoid redundant computations.

```python
import streamlit as st
import pandas as pd
import time

def expensive_process(option, add):
    with st.spinner('Processing...'):
        time.sleep(5)
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C':[7, 8, 9]}) + add
    return (df, add)

cols = st.columns(2)
option = cols[0].selectbox('Select a number', options=['1', '2', '3'])
add = cols[1].number_input('Add a number', min_value=0, max_value=10)

if 'processed' not in st.session_state:
    st.session_state.processed = {}

# Process and save results
if st.button('Process'):
    result = expensive_process(option, add)
    st.session_state.processed[option] = result
    st.write(f'Option {option} processed with add {add}')
    result[0]
```

--------------------------------

### Streamlit Fragment: Triggering Rerun with st.rerun

Source: https://docs.streamlit.io/develop/api-reference/execution-flow/st

Shows how to trigger an app rerun from within a fragment by calling `st.rerun`. This example increments a click counter and reruns the app when the count reaches a multiple of five.

```python
import streamlit as st

if "clicks" not in st.session_state:
    st.session_state.clicks = 0

@st.fragment
def count_to_five():
    if st.button("Plus one!"):
        st.session_state.clicks += 1
        if st.session_state.clicks % 5 == 0:
            st.rerun()
    return

count_to_five()
st.header(f"Multiples of five clicks: {st.session_state.clicks // 5}")

if st.button("Check click count"):
    st.toast(f"## Total clicks: {st.session_state.clicks}")
```

--------------------------------

### Streamlit Widget Object Comparison Example

Source: https://docs.streamlit.io/develop/concepts/design/custom-classes

Illustrates a pathological example where comparing instances of different classes, even if they appear identical, can lead to unexpected results due to how Streamlit stores and retrieves widget options from Session State. This occurs when class identity is checked within comparison operators.

```python
import streamlit as st
from dataclasses import dataclass

@dataclass
class Student:
    student_id: int
    name: str

Marshall_A = Student(1, "Marshall")
if "B" not in st.session_state:
    st.session_state.B = Student(1, "Marshall")
Marshall_B = st.session_state.B

options = [Marshall_A,Marshall_B]
selected = st.selectbox("Pick", options)

# This comparison does not return expected results:
# selected == Marshall_A
# This comparison evaluates as expected:
# selected == Marshall_B
```

--------------------------------

### Install GCSFS and FilesConnection for Streamlit

Source: https://docs.streamlit.io/develop/tutorials/databases/gcs

This command-line snippet shows how to add the necessary packages to your Streamlit project's `requirements.txt` file. `gcsfs` provides the interface for interacting with Google Cloud Storage, and `st-files-connection` enables Streamlit's file connection capabilities. Pinning versions is recommended for reproducible builds.

```bash
# requirements.txt
gcsfs==x.x.x
st-files-connection

```

--------------------------------

### Run Streamlit App from Terminal

Source: https://docs.streamlit.io/develop/tutorials/elements/annotate-an-altair-chart

This command initiates a Streamlit application by running the specified Python script (`app.py`). It requires Streamlit to be installed and assumes the script is in the current directory. The command opens the app in your default web browser.

```bash
streamlit run app.py

```

--------------------------------

### Streamlit App Initialization: Import Libraries

Source: https://docs.streamlit.io/develop/tutorials/execution-flow/start-and-stop-fragment-auto-reruns

This Python code snippet initializes a Streamlit application by importing necessary libraries: `streamlit` for app development, `pandas` for data manipulation, `numpy` for numerical operations, and `datetime` and `timedelta` for handling time-based data. These libraries are fundamental for creating data-driven Streamlit applications.

```Python
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
```

--------------------------------

### Display Code with Streamlit Echo

Source: https://docs.streamlit.io/develop/quick-reference/cheat-sheet

Demonstrates the use of `st.echo()` to display code that will be executed and its output printed to the application. This is helpful for showing examples or debugging.

```python
with st.echo():
    st.write("Code will be executed and printed")
```

--------------------------------

### Control Widget Disabled State with Streamlit Toggle Button

Source: https://docs.streamlit.io/develop/concepts/design/buttons

This example demonstrates an alternative way to implement a toggle button effect. Instead of conditionally displaying widgets, it uses the button's state to control the `disabled` parameter of another widget, such as st.slider.

```python
import streamlit as st

if 'button' not in st.session_state:
    st.session_state.button = False

def click_button():
    st.session_state.button = not st.session_state.button

st.button('Click me', on_click=click_button)

st.slider('Select a value', disabled=st.session_state.button)

```

--------------------------------

### Create TiDB Database and Table (SQL)

Source: https://docs.streamlit.io/develop/tutorials/databases/tidb

SQL commands to create a new database named 'pets', select it, define a table 'mytable' with 'name' and 'pet' columns, and insert sample data. This sets up the necessary structure in TiDB for the Streamlit application.

```sql
CREATE DATABASE pets;

USE pets;

CREATE TABLE mytable (
    name            varchar(80),
    pet             varchar(80)
);

INSERT INTO mytable VALUES ('Mary', 'dog'), ('John', 'cat'), ('Robert', 'bird');
```

--------------------------------

### Third-Party Components

Source: https://docs.streamlit.io/develop/api-reference

Examples of third-party Streamlit components for enhanced functionality.

```APIDOC
## st_tags (by @gagan3012)

### Description
Add tags to your Streamlit apps.

### Method
Python

### Endpoint
N/A

### Parameters
None

### Request Example
```python
st_tags(label='# Enter Keywords:', text='Press enter to add more', value=['Zero', 'One', 'Two'], suggestions=['five', 'six', 'seven', 'eight', 'nine', 'three', 'eleven', 'ten', 'four'], maxtags = 4, key='1')
```

### Response
None

## NLU (by @JohnSnowLabs)

### Description
Apply text mining on a dataframe.

### Method
Python

### Endpoint
N/A

### Parameters
None

### Request Example
```python
nlu.load("sentiment").predict("I love NLU! <3")
```

### Response
None

## mention (by @arnaudmiribel)

### Description
Create a mention link with an icon and URL.

### Method
Python

### Endpoint
N/A

### Parameters
None

### Request Example
```python
mention(label="An awesome Streamlit App", icon="streamlit",  url="https://extras.streamlit.app",)
```

### Response
None

## annotated_text (by @tvst)

### Description
Display annotated text in Streamlit apps.

### Method
Python

### Endpoint
N/A

### Parameters
None

### Request Example
```python
annotated_text("This ", ("is", "verb"), " some ", ("annotated", "adj"), ("text", "noun"), " for those of ", ("you", "pronoun"), " who ", ("like", "verb"), " this sort of ", ("thing", "noun"), ".")
```

### Response
None

## st_canvas (by @andfanilo)

### Description
Provides a sketching canvas using Fabric.js.

### Method
Python

### Endpoint
N/A

### Parameters
None

### Request Example
```python
st_canvas(fill_color="rgba(255, 165, 0, 0.3)", stroke_width=stroke_width, stroke_color=stroke_color, background_color=bg_color, background_image=Image.open(bg_image) if bg_image else None, update_streamlit=realtime_update, height=150, drawing_mode=drawing_mode, point_display_radius=point_display_radius if drawing_mode == 'point' else 0, key="canvas",)
```

### Response
None
```

--------------------------------

### Initialize Streamlit App in Current Directory

Source: https://docs.streamlit.io/develop/api-reference/cli/init

Creates a new Streamlit project structure in the current working directory. This includes a `requirements.txt` and `streamlit_app.py`. After creation, it prompts to run the app.

```bash
streamlit init

```

```text
CWD/
├── requirements.txt
└── streamlit_app.py

```

--------------------------------

### Change Streamlit App Title

Source: https://docs.streamlit.io/get-started/installation/anaconda-distribution

Modifies the 'Hello World' Streamlit app to display a title using `st.title` instead of general text with `st.write`.

```python
import streamlit as st

st.title("Hello World")
```

--------------------------------

### Modify Session State with Button Callbacks in Streamlit

Source: https://docs.streamlit.io/develop/concepts/design/buttons

This example uses button callbacks to modify `st.session_state` in Streamlit. Callbacks are executed before the script reruns, ensuring that the state is updated consistently regardless of the button's position in the script. Dependencies include Streamlit and Pandas.

```python
import streamlit as st
import pandas as pd

if 'name' not in st.session_state:
    st.session_state['name'] = 'John Doe'

def change_name(name):
    st.session_state['name'] = name

st.header(st.session_state['name'])

st.button('Jane', on_click=change_name, args=['Jane Doe'])
st.button('John', on_click=change_name, args=['John Doe'])

st.header(st.session_state['name'])
```

--------------------------------

### Modify Session State with Buttons in Streamlit (Direct Modification)

Source: https://docs.streamlit.io/develop/concepts/design/buttons

This example shows how clicking buttons directly modifies a value in `st.session_state`. It illustrates a potential issue where the displayed state before the buttons might lag behind the state after the buttons due to the script rerun order. Dependencies include Streamlit and Pandas.

```python
import streamlit as st
import pandas as pd

if 'name' not in st.session_state:
    st.session_state['name'] = 'John Doe'

st.header(st.session_state['name'])

if st.button('Jane'):
    st.session_state['name'] = 'Jane Doe'

if st.button('John'):
    st.session_state['name'] = 'John Doe'

st.header(st.session_state['name'])
```

--------------------------------

### Apply Streamlit Extras Styling

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=configuration&slug=data

Applies various utility features from the streamlit_extras library, such as styling metric cards. This example demonstrates styling metric cards.

```python
from streamlit_extras.metric_cards import style_metric_cards
import streamlit as st

# Assuming col3 is a Streamlit column object
# col3.metric(label="No Change", value=5000, delta=0)

style_metric_cards()
```

--------------------------------

### Build Custom Connection with BaseConnection in Python

Source: https://docs.streamlit.io/develop/api-reference/connections

Demonstrates how to create a custom Streamlit connection by subclassing `BaseConnection`. This involves implementing the `_connect` method to establish the actual connection and a `query` method for data retrieval. It allows for integration with custom data sources or APIs.

```python
class MyConnection(BaseConnection[myconn.MyConnection]):
    def _connect(self, **kwargs) -> MyConnection:
        return myconn.connect(**self._secrets, **kwargs)
    def query(self, query):
        return self._instance.query(query)
```

--------------------------------

### AppTest.get

Source: https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1

Get elements or widgets of a specified type from the current page.

```APIDOC
## AppTest.get

### Description
Get elements or widgets of the specified type. This method returns the collection of all elements or widgets of the specified type on the current page. Retrieve a specific element by using its index (order on page) or key lookup.

### Method
GET

### Endpoint
/websites/streamlit_io/AppTest/get

### Parameters
#### Query Parameters
- **element_type** (str) - Required - An element attribute of `AppTest`. For example, "button", "caption", or "chat_input".

### Response
#### Success Response (200)
- **Sequence of Elements** (Sequence) - Sequence of elements of the given type. Individual elements can be accessed from a Sequence by index (order on the page). When getting and `element_type` that is a widget, individual widgets can be accessed by key. For example, `at.get("text")[0]` for the first `st.text` element or `at.get("slider")(key="my_key")` for the `st.slider` widget with a given key.

#### Response Example
```json
{
  "elements": [
    {
      "type": "text",
      "content": "Some text content"
    },
    {
      "type": "button",
      "label": "Click Me"
    }
  ]
}
```
```

--------------------------------

### Deactivate Virtual Environment

Source: https://docs.streamlit.io/get-started/installation/command-line

Exit the active virtual environment and return to your system's default shell. This command is used when you are finished working on the project.

```bash
deactivate
```

--------------------------------

### Streamlit CLI: Version Information

Source: https://docs.streamlit.io/develop/api-reference/cli

Command to display the currently installed version of the Streamlit library.

```bash
streamlit version
```

--------------------------------

### Streamlit Vega-Lite Chart Integration

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

Displays a chart created using the Vega-Lite specification within Streamlit. Requires Vega-Lite to be installed.

```python
import streamlit as st

# Assuming my_vega_lite_chart is a Vega-Lite chart specification
# st.vega_lite_chart(my_vega_lite_chart)
```

--------------------------------

### Get Configuration Option

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=configuration&slug=chat

Retrieves a specific configuration option from Streamlit's settings.

```APIDOC
## GET /api/config/option

### Description
Retrieve a single configuration option.

### Method
GET

### Endpoint
`/api/config/option`

### Parameters
#### Query Parameters
- **key** (string) - Required - The key of the configuration option to retrieve (e.g., `theme.primaryColor`).

### Request Example
```python
st.get_option("theme.primaryColor")
```

### Response
#### Success Response (200)
- **value** (any) - The value of the requested configuration option.
```

--------------------------------

### Streamlit Multiselect with Enum Example

Source: https://docs.streamlit.io/develop/concepts/design/custom-classes

Shows how to use Python's Enum class as options for st.multiselect. It highlights a potential issue in older Streamlit versions (pre-1.29.0) where Enum classes redefined on script reruns could cause comparison failures. The example demonstrates the expected behavior with enumCoercion enabled.

```python
from enum import Enum
import streamlit as st

# class syntax
class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

selected_colors = set(st.multiselect("Pick colors", options=Color))

if selected_colors == {Color.RED, Color.GREEN}:
    st.write("Hooray, you found the color YELLOW!")
```

--------------------------------

### Load ML Model with Hugging Face Transformers

Source: https://docs.streamlit.io/develop/concepts/architecture/caching

Loads a sentiment analysis model using Hugging Face's transformers library. This is a basic example before caching is applied.

```python
from transformers import pipeline
model = pipeline("sentiment-analysis")  # 👇 Load the model
```

--------------------------------

### Install google-cloud-bigquery for Streamlit

Source: https://docs.streamlit.io/develop/tutorials/databases/bigquery

This command adds the google-cloud-bigquery library to your Streamlit project's dependencies. Pinning the version is recommended for reproducible builds. This package is essential for interacting with BigQuery from your Python application.

```bash
# requirements.txt
google-cloud-bigquery==x.x.x

```

--------------------------------

### Layout with Columns and Radio Buttons in Streamlit

Source: https://docs.streamlit.io/get-started/fundamentals/main-concepts

Demonstrates using `st.columns` to arrange widgets side-by-side and `st.radio` within a column. This example shows how to place a button in one column and a radio button group with text display in another.

```python
import streamlit as st

left_column, right_column = st.columns(2)
# You can use a column just like st.sidebar:
left_column.button('Press me!')

# Or even better, call Streamlit functions inside a "with" block:
with right_column:
    chosen = st.radio(
        'Sorting hat',
        ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
    st.write(f"You are in {chosen} house!")
```

--------------------------------

### Configure Auth0 Client Keywords in Streamlit

Source: https://docs.streamlit.io/develop/concepts/connections/authentication

This TOML configuration snippet demonstrates how to set up authentication for Auth0 within Streamlit. It includes essential details like redirect URI, cookie secret, client ID, client secret, and server metadata URL. Crucially, it shows how to pass custom client keywords, such as 'prompt' set to 'login', to modify the authentication flow.

```toml
[auth]
redirect_uri = "http://localhost:8501/oauth2callback"
cookie_secret = "xxx"

[auth.auth0]
client_id = "xxx"
client_secret = "xxx"
server_metadata_url = "https://{account}.{region}.auth0.com/.well-known/openid-configuration"
client_kwargs = { "prompt" = "login" }
```

--------------------------------

### Run Streamlit Navigation

Source: https://docs.streamlit.io/develop/tutorials/multipage/dynamic-navigation

Executes the Streamlit navigation object, which renders the navigation menu and the currently selected page. This is the final step in displaying the app's structure.

```python
pg.run()
```

--------------------------------

### Streamlit Altair Chart Integration

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

Displays a chart created using the Altair visualization library within Streamlit. Requires Altair to be installed.

```python
import streamlit as st
import altair as alt

# Assuming my_altair_chart is an Altair chart object
# st.altair_chart(my_altair_chart)
```

--------------------------------

### Initialize Streamlit App

Source: https://docs.streamlit.io/develop/tutorials/authentication/google

This Python code snippet initializes a basic Streamlit application. It imports the Streamlit library, which is necessary for building interactive web applications with Python. This is the starting point for adding more complex Streamlit components and logic.

```python
import streamlit as st
```

--------------------------------

### Initialize Streamlit Page Dictionary

Source: https://docs.streamlit.io/develop/tutorials/multipage/dynamic-navigation

Initializes an empty dictionary to store page lists. This dictionary will be populated based on the user's role to create a dynamic navigation menu.

```python
page_dict = {}
```

--------------------------------

### Streamlit Login Page Test Cases (Python)

Source: https://docs.streamlit.io/develop/concepts/app-testing/examples

Unit tests for the Streamlit login page application using `streamlit.testing.v1.AppTest`. These tests cover scenarios like no interaction, incorrect password, correct password, and logging out. Dummy secrets are set before each test run.

```Python
from streamlit.testing.v1 import AppTest

def test_no_interaction():
    at = AppTest.from_file("app.py")
    at.secrets["password"] = "streamlit"
    at.run()
    assert at.session_state["status"] == "unverified"
    assert len(at.text_input) == 1
    assert len(at.warning) == 0
    assert len(at.success) == 0
    assert len(at.button) == 0
    assert at.text_input[0].value == ""

def test_incorrect_password():
    at = AppTest.from_file("app.py")
    at.secrets["password"] = "streamlit"
    at.run()
    at.text_input[0].input("balloon").run()
    assert at.session_state["status"] == "incorrect"
    assert len(at.text_input) == 1
    assert len(at.warning) == 1
    assert len(at.success) == 0
    assert len(at.button) == 0
    assert at.text_input[0].value == ""
    assert "Incorrect password" in at.warning[0].value

def test_correct_password():
    at = AppTest.from_file("app.py")
    at.secrets["password"] = "streamlit"
    at.run()
    at.text_input[0].input("streamlit").run()
    assert at.session_state["status"] == "verified"
    assert len(at.text_input) == 0
    assert len(at.warning) == 0
    assert len(at.success) == 1
    assert len(at.button) == 1
    assert "Login successful" in at.success[0].value
    assert at.button[0].label == "Log out"

def test_log_out():
    at = AppTest.from_file("app.py")
    at.secrets["password"] = "streamlit"
    at.session_state["status"] = "verified"
    at.run()
    at.button[0].click().run()
    assert at.session_state["status"] == "unverified"
    assert len(at.text_input) == 1
    assert len(at.warning) == 0
    assert len(at.success) == 0
    assert len(at.button) == 0
    assert at.text_input[0].value == ""
```

--------------------------------

### Configure Local Secrets with TOML

Source: https://docs.streamlit.io/develop/concepts/connections/secrets-management

Example of a TOML file for configuring secrets locally. This file should be placed in `~/.streamlit/secrets.toml` (global) or `$CWD/.streamlit/secrets.toml` (per-project) and added to `.gitignore`. It supports direct key-value pairs and nested sections.

```toml
# Everything in this section will be available as an environment variable
db_username = "Jane"
db_password = "mypassword"

# You can also add other sections if you like.
# The contents of sections as shown below will not become environment variables,
# but they'll be easily accessible from within Streamlit anyway as we show
# later in this doc.
[my_other_secrets]
things_i_like = ["Streamlit", "Python"]
```

--------------------------------

### Modify Session State with Buttons and st.rerun in Streamlit

Source: https://docs.streamlit.io/develop/concepts/design/buttons

This example demonstrates modifying `st.session_state` using buttons and explicitly calling `st.rerun()`. This approach ensures that changes made by the button are reflected immediately, even if the state is accessed before the button in the script. This method causes the script to rerun twice when a button is clicked. Dependencies include Streamlit and Pandas.

```python
import streamlit as st
import pandas as pd

if 'name' not in st.session_state:
    st.session_state['name'] = 'John Doe'

st.header(st.session_state['name'])

if st.button('Jane'):
    st.session_state['name'] = 'Jane Doe'
    st.rerun()

if st.button('John'):
    st.session_state['name'] = 'John Doe'
    st.rerun()

st.header(st.session_state['name'])
```

--------------------------------

### Streamlit Plotting Demo with Animation

Source: https://docs.streamlit.io/get-started/tutorials/create-a-multipage-app

This demo illustrates plotting and animation using Streamlit. It generates random numbers in a loop for approximately 5 seconds, updating a progress bar and status text in the sidebar. Requires numpy for random number generation. The code snippet shows the setup for the progress bar and initial data.

```python
import streamlit as st
import time
import numpy as np

def plotting_demo():
    import streamlit as st
    import time
    import numpy as np

    st.markdown(f'# {list(page_names_to_funcs.keys())[1]}')
    st.write(
        """
        This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!
"""
    )

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    last_rows = np.random.randn(1, 1)
    
```

--------------------------------

### Use an Empty Container for Dynamic Updates in Streamlit

Source: https://docs.streamlit.io/develop/api-reference/layout

This example shows how to use `st.empty()` to create a placeholder that can be later replaced with new content. This is useful for dynamic updates within the Streamlit app.

```python
c = st.empty()
st.write("This will show last")
c.write("This will be replaced")
c.write("This will show first")
```

--------------------------------

### Run Streamlit Hello App with Custom Theme Color (Python)

Source: https://docs.streamlit.io/develop/api-reference/cli/hello

Runs the Streamlit 'Hello' app with a custom theme color, such as blue. This demonstrates how to apply configuration options using the `--<section>.<option>=<value>` format.

```python
streamlit hello --theme.primaryColor=blue
```

--------------------------------

### Streamlit Graphviz Chart Integration

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

Displays a graph using the Graphviz specification within Streamlit. Requires Graphviz to be installed.

```python
import streamlit as st

# Assuming my_graphviz_spec is a Graphviz DOT string or dictionary
# st.graphviz_chart(my_graphviz_spec)
```

--------------------------------

### Add PostgreSQL Dependencies to requirements.txt

Source: https://docs.streamlit.io/develop/tutorials/databases/postgresql

Specifies the necessary Python packages, psycopg2-binary and SQLAlchemy, along with their versions, to be installed for Streamlit to connect to a PostgreSQL database. This file is used by Streamlit Community Cloud for deployment.

```text
# requirements.txt
psycopg2-binary==x.x.x
saltar==x.x.x
```

--------------------------------

### Snowflake Connection Configuration (TOML)

Source: https://docs.streamlit.io/develop/tutorials/databases/snowflake

Example TOML configuration for Snowflake connection parameters including account identifier. Note that the account identifier format requires hyphens for the Snowflake Connector for Python.

```toml
role = "ACCOUNTADMIN"
warehouse = "COMPUTE_WH"
database = "PETS"
schema = "PUBLIC"

account = "xxxxxxx-xxxxxxx"
```

--------------------------------

### Streamlit: Advanced SQLConnection with Snowflake using Python kwargs

Source: https://docs.streamlit.io/develop/concepts/connections/connecting-to-data

This example demonstrates configuring a Streamlit SQLConnection for Snowflake directly within the Python script using keyword arguments (`**kwargs`). It bypasses the need for a secrets file by providing the connection URL and `connect_args` directly to the `st.connection` call. This method allows for dynamic configuration or when secrets files are not preferred.

```python
# streamlit_app.py

import streamlit as st

# secrets.toml is not needed
conn = st.connection(
    "snowflake",
    "sql",
    url = "snowflake://<user_login_name>@<account_identifier>/",
    connect_args = dict(
        authenticator = "externalbrowser",
        warehouse = "xxx",
        role = "xxx",
    )
)
# ...
```

--------------------------------

### Running a Streamlit Multipage App

Source: https://docs.streamlit.io/develop/concepts/multipage-apps/pages-directory

To run a multipage Streamlit app structured with the `pages/` directory, use the `streamlit run` command followed by the entrypoint file. This command initiates the app, and Streamlit automatically handles the navigation based on the `pages/` directory content.

```bash
streamlit run your_homepage.py
```

--------------------------------

### Streamlit PyDeck Chart Integration

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

Displays a chart using the PyDeck visualization library within Streamlit. Requires PyDeck to be installed.

```python
import streamlit as st
import pydeck as pdk

# Assuming my_pydeck_chart is a PyDeck layer or deck object
# st.pydeck_chart(my_pydeck_chart)
```

--------------------------------

### Specify Python Dependencies for Streamlit App

Source: https://docs.streamlit.io/develop/tutorials/authentication/google

This snippet shows how to define the necessary Python dependencies for a Streamlit application to be deployed on Community Cloud. It ensures that Streamlit and Authlib are installed with specific versions.

```txt
streamlit>=1.42.0
Authlib>=1.3.2
```

--------------------------------

### Add pyTigerGraph Dependency

Source: https://docs.streamlit.io/develop/tutorials/databases/tigergraph

Specifies the pyTigerGraph package to be installed for the Streamlit application. It's recommended to pin the version for consistent deployments.

```text
# requirements.txt
pyTigerGraph==x.x.x
```

--------------------------------

### Run Streamlit App with Configuration Options

Source: https://docs.streamlit.io/develop/api-reference/cli/run

Starts a Streamlit app while setting specific configuration options. This allows customization of app behavior and appearance, such as theme colors or error display.

```bash
streamlit run your_app.py --client.showErrorDetails=False --theme.primaryColor=blue
```

```bash
streamlit run --client.showErrorDetails=False --theme.primaryColor=blue
```

--------------------------------

### Get Streamlit Configuration Option

Source: https://docs.streamlit.io/develop/api-reference/configuration

Demonstrates how to retrieve a specific configuration option from the Streamlit settings. This function is useful for accessing predefined or custom settings within your application.

```python
st.get_option("theme.primaryColor")
```

--------------------------------

### Streamlit Extras: Metric Cards

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=configuration&slug=layouts

Provides enhanced metric cards for Streamlit, including styling options. This example shows how to display a metric and then apply card styling. Requires streamlit-extras.

```python
from streamlit_extras.metric_cards import style_metric_cards
import streamlit as st

# Assuming 'col3' is a Streamlit column
col3 = st.columns(1)[0]
col3.metric(label="No Change", value=5000, delta=0)

style_metric_cards()
```

--------------------------------

### Image Coordinates

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=configuration&slug=data

Gets the coordinates of clicks on an image displayed in Streamlit. Created by @blackary.

```APIDOC
## Image Coordinates

### Description
Get the coordinates of clicks on an image. Created by @blackary.

### Method
N/A (This is a component usage example)

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
from streamlit_image_coordinates import streamlit_image_coordinates
value = streamlit_image_coordinates("https://placekitten.com/200/300")

st.write(value)
```

### Response
#### Success Response (200)
N/A (Component interaction)

#### Response Example
N/A
```

--------------------------------

### Run Streamlit App (Subdirectory with Specific File)

Source: https://docs.streamlit.io/develop/api-reference/cli/run

Starts a Streamlit app from a specific file within a subdirectory. This provides flexibility in organizing your app files.

```bash
streamlit run your_subdirectory/your_app.py
```

--------------------------------

### Initialize Streamlit Connection to SQL Server

Source: https://docs.streamlit.io/develop/tutorials/databases/mssql

Python code snippet to import necessary libraries (streamlit, pyodbc) and initialize a database connection. This is the starting point for a Streamlit app interacting with SQL Server.

```python
import streamlit as st
import pyodbc

# Initialize connection.

```

--------------------------------

### Streamlit Matplotlib Integration

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

Renders a Matplotlib figure within a Streamlit application. Requires Matplotlib to be installed.

```python
import streamlit as st
import matplotlib.pyplot as plt

# Assuming my_mpl_figure is a matplotlib.pyplot figure object
# st.pyplot(my_mpl_figure)
```

--------------------------------

### Display Static Tables with st.table()

Source: https://docs.streamlit.io/get-started/fundamentals/main-concepts

Demonstrates how to display a Pandas DataFrame as a static table using st.table(). This is useful when an interactive table is not desired and a simple, fixed representation is preferred.

```python
import streamlit as st
import numpy as np
import pandas as pd

dataframe = pd.DataFrame(
    np.random.randn(10, 20),
    columns=('col %d' % i for i in range(20)))
st.table(dataframe)
```

--------------------------------

### Streamlit Media Display Functions

Source: https://docs.streamlit.io/develop/quick-reference/cheat-sheet

Provides examples of Streamlit functions for displaying various media types such as images, logos, PDFs, audio, and video. Includes options for video subtitles.

```python
st.image("./header.png")
st.logo("logo.jpg")
st.pdf("my_document.pdf")
st.audio(data)
st.video(data)
st.video(data, subtitles="./subs.vtt")
```

--------------------------------

### Streamlit Extras for Annotations

Source: https://docs.streamlit.io/develop/api-reference_slug=%5B...slug%5D

Adds annotations to charts using the streamlit-extras library. This example shows how to add date-based annotations to a chart.

```python
import streamlit as st
# Assuming 'chart' is an existing chart object (e.g., Altair)
# Assuming 'get_annotations_chart' is a function from streamlit_extras
# from streamlit_extras.chart_annotations import get_annotations_chart

# Placeholder for chart object and function for demonstration
chart = None 
def get_annotations_chart(annotations):
    return annotations # Dummy return

chart = get_annotations_chart(annotations=[
    ("Mar 01, 2008", "Pretty good day for GOOG"), 
    ("Dec 01, 2007", "Something's going wrong for GOOG & AAPL"), 
    ("Nov 01, 2008", "Market starts again thanks to..."), 
    ("Dec 01, 2009", "Small crash for GOOG after..."),
])

st.altair_chart(chart, use_container_width=True)
```

--------------------------------

### Initialize AppTest from File

Source: https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1

Initializes an AppTest instance from a Python script file. This is the recommended method for setting up tests. It takes the script path as an argument and optionally accepts default timeout, arguments, and keyword arguments for the app.

```python
app = st.testing.v1.AppTest.from_file("my_app.py")
```

--------------------------------

### Streamlit: Advanced SQLConnection with Snowflake using Secrets File

Source: https://docs.streamlit.io/develop/concepts/connections/connecting-to-data

This example shows how to configure a Streamlit SQLConnection for Snowflake using a secrets file. It includes specifying the connection URL and passing additional `create_engine_kwargs`, such as `authenticator`, `warehouse`, and `role`, which are necessary for certain SQLAlchemy dialects like Snowflake.

```toml
# .streamlit/secrets.toml

[connections.snowflake]
url = "snowflake://<user_login_name>@<account_identifier>/"

[connections.snowflake.create_engine_kwargs.connect_args]
authenticator = "externalbrowser"
warehouse = "xxx"
role = "xxx"
```

```python
# streamlit_app.py

import streamlit as st

# url and connect_args from secrets.toml above are picked up and used here
conn = st.connection("snowflake", "sql")
# ...
```

--------------------------------

### Display DataFrame using Streamlit Magic Command

Source: https://docs.streamlit.io/get-started/fundamentals/main-concepts

Demonstrates Streamlit's 'magic' feature where a variable or literal on its own line is automatically written to the app. This example displays a Pandas DataFrame.

```python
"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd
df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

df
```

--------------------------------

### Third-Party Components

Source: https://docs.streamlit.io/develop/api-reference_slug=private-gsheet

Streamlit supports third-party components developed by the community. Examples include tags, NLU, and annotated text.

```APIDOC
## Third-Party Components

### Description
Streamlit allows integration with third-party components created by the community. These components extend Streamlit's functionality.

### Examples

#### st_tags
**Description:** Add tags to your Streamlit apps.
**Created by:** @gagan3012
**Python Example:**
```python
st_tags(label='# Enter Keywords:', text='Press enter to add more', value=['Zero', 'One', 'Two'], suggestions=['five', 'six', 'seven', 'eight', 'nine', 'three', 'eleven', 'ten', 'four'], maxtags = 4, key='1')
```

#### nlu
**Description:** Apply text mining on a dataframe.
**Created by:** @JohnSnowLabs
**Python Example:**
```python
nlu.load("sentiment").predict("I love NLU! <3")
```

#### mention (from Streamlit Extras)
**Description:** A library with useful Streamlit extras.
**Created by:** @arnaudmiribel
**Python Example:**
```python
mention(label="An awesome Streamlit App", icon="streamlit",  url="https://extras.streamlit.app",)
```

#### annotated_text
**Description:** Display annotated text in Streamlit apps.
**Created by:** @tvst
**Python Example:**
```python
annotated_text("This ", ("is", "verb"), " some ", ("annotated", "adj"), ("text", "noun"), " for those of ", ("you", "pronoun"), " who ", ("like", "verb"), " this sort of ", ("thing", "noun"), ".")
```

#### st_canvas
**Description:** Provides a sketching canvas using Fabric.js.
**Created by:** @andfanilo
**Python Example:**
```python
st_canvas(fill_color="rgba(255, 165, 0, 0.3)", stroke_width=stroke_width, stroke_color=stroke_color, background_color=bg_color, background_image=Image.open(bg_image) if bg_image else None, update_streamlit=realtime_update, height=150, drawing_mode=drawing_mode, point_display_radius=point_display_radius if drawing_mode == 'point' else 0, key="canvas",)
```

### Response
#### Success Response (200)
- **component_name** (string) - The name of the third-party component.
- **status** (string) - Indicates successful integration or rendering.

#### Response Example
```json
{
  "component_name": "st_tags",
  "status": "integrated"
}
```
```

--------------------------------

### Streamlit Plotly Chart Integration

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

Displays an interactive Plotly chart within Streamlit. Requires Plotly to be installed.

```python
import streamlit as st
import plotly.graph_objects as go

# Assuming my_plotly_chart is a Plotly figure object
# st.plotly_chart(my_plotly_chart)
```

--------------------------------

### Create Streamlit Navigation for Login Page

Source: https://docs.streamlit.io/develop/tutorials/multipage/dynamic-navigation

Creates a Streamlit navigation menu containing only the login page when the user is not logged in or has no access to other pages.

```python
else:
    pg = st.navigation([st.Page(login)])
```

--------------------------------

### Filename to Label Conversion Examples

Source: https://docs.streamlit.io/st.page

Illustrates how Streamlit converts Python filenames into user-friendly page labels for the navigation menu. Underscores are treated as spaces, and numerical prefixes control order. This applies to both `pages/` directory and `st.Page` usage.

```python
# Filenames and callables that display as "Awesome page" in navigation:
# "Awesome page.py"
# "Awesome_page.py"
# "02Awesome_page.py"
# "--Awesome_page.py"
# "1_Awesome_page.py"
# "33 - Awesome page.py"
# Awesome_page()
# _Awesome_page()
# __Awesome_page__()
```

--------------------------------

### Handle Nonexistent Secret Key Error (Python)

Source: https://docs.streamlit.io/develop/concepts/connections/secrets-management

Shows how Streamlit raises a `KeyError` when attempting to access a secret that does not exist in the `secrets.toml` file. This example demonstrates the error scenario.

```python
import streamlit as st

st.write(st.secrets["nonexistent_key"])
```

--------------------------------

### Create Interactive Plots with Plost

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=configuration&slug=status

This snippet demonstrates using 'plost', a plotting library for Streamlit, to create interactive charts. It shows an example of a line chart with time, stock value, and stock name as parameters.

```python
import streamlit as st
import plost

# Assuming 'my_dataframe' is a Pandas DataFrame
# plost.line_chart(my_dataframe, x='time', y='stock_value', color='stock_name')
```

--------------------------------

### Configure Streamlit Column Display

Source: https://docs.streamlit.io/develop/api-reference/data

Defines configuration options for columns in Streamlit dataframes and data editors. This example shows how to configure a number column with specific formatting and constraints.

```python
st.column_config.NumberColumn("Price (in USD)", min_value=0, format="$%d")
```

--------------------------------

### Streamlit Multipage App Navigation

Source: https://docs.streamlit.io/get-started/tutorials/create-a-multipage-app

Illustrates how to implement multipage navigation in a Streamlit application using a sidebar selectbox. Each page is defined as a function, and the selectbox dynamically calls the selected function. This approach allows for better code organization and scalability. Requires `streamlit`.

```python
import streamlit as st

# Assume intro, plotting_demo, mapping_demo, data_frame_demo are defined elsewhere
# For example:
def intro():
    st.markdown("# Welcome!")
def plotting_demo():
    st.markdown("# Plotting Demo")
def mapping_demo():
    st.markdown("# Mapping Demo")
def data_frame_demo():
    st.markdown("# DataFrame Demo")

page_names_to_funcs = {
    "—": intro,
    "Plotting Demo": plotting_demo,
    "Mapping Demo": mapping_demo,
    "DataFrame Demo": data_frame_demo
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
```

--------------------------------

### Build Chat-Based Apps with Streamlit

Source: https://docs.streamlit.io/develop/quick-reference/cheat-sheet

Shows how to create chat-based applications using Streamlit's chat message container and chat input widgets. It includes examples of displaying user messages, charts, and handling user input.

```python
# Insert a chat message container.
with st.chat_message("user"):
    st.write("Hello 👋")
    st.line_chart(np.random.randn(30, 3))

# Display a chat input widget at the bottom of the app.
st.chat_input("Say something")

# Display a chat input widget inline.
with st.container():
    st.chat_input("Say something")
```

--------------------------------

### Initialize Streamlit App and Import Libraries in Python

Source: https://docs.streamlit.io/develop/tutorials/elements/annotate-an-altair-chart

This Python snippet sets up a basic Streamlit application by importing necessary libraries: Streamlit for the web app framework, Altair for charting, Pandas for data manipulation, and Vega-datasets for sample data. This forms the foundation for building interactive data visualizations.

```python
import streamlit as st
import altair as alt
import pandas as pd
from vega_datasets import data

```

--------------------------------

### Display ECharts in Streamlit

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=configuration&slug=status

This example shows how to render charts using the ECharts library within a Streamlit application. It requires an 'options' object defining the ECharts configuration.

```python
from streamlit_echarts import st_echarts

# Assuming 'options' is a dictionary containing ECharts configurations
# st_echarts(options=options)
```

--------------------------------

### Streamlit Counter App with Session State and Callbacks

Source: https://docs.streamlit.io/develop/concepts/architecture/session-state

A basic Streamlit application demonstrating a counter that increments using session state and a callback function triggered by a button click. It initializes the session state if 'count' is not present and defines a function to increment the counter.

```python
import streamlit as st

st.title('Counter Example using Callbacks')
if 'count' not in st.session_state:
    st.session_state.count = 0

def increment_counter():
    st.session_state.count += 1

st.button('Increment', on_click=increment_counter)

st.write('Count = ', st.session_state.count)
```

--------------------------------

### Example Python Script for Streamlit App

Source: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies

A basic Streamlit Python script demonstrating the use of standard libraries and common data science packages like pandas and numpy. This script serves as a reference for identifying dependencies that need to be declared.

```python
import streamlit as st
import pandas as pd
import numpy as np
import math
import random

st.write("Hi!")
```

--------------------------------

### Define Streamlit Respond and Admin Pages

Source: https://docs.streamlit.io/develop/tutorials/multipage/dynamic-navigation

Defines Streamlit pages for responding to requests and administrative tasks. It also includes dynamic default page settings based on user roles.

```python
respond_1 = st.Page(
    "respond/respond_1.py",
    title="Respond 1",
    icon=":material/healing:",
    default=(role == "Responder"),
)
respond_2 = st.Page(
    "respond/respond_2.py", title="Respond 2", icon=":material/handyman:"
)
admin_1 = st.Page(
    "admin/admin_1.py",
    title="Admin 1",
    icon=":material/person_add:",
    default=(role == "Admin"),
)
admin_2 = st.Page("admin/admin_2.py", title="Admin 2", icon=":material/security:")
```

--------------------------------

### Download Files in Streamlit using st.download_button

Source: https://docs.streamlit.io/knowledge-base/using-streamlit/how-download-file-streamlit

This Python code demonstrates how to use Streamlit's `st.download_button` widget to enable file downloads. It covers downloading text files (like CSV) and binary files (like ZIP) by providing content directly or by referencing file objects. The examples show default behavior and custom file naming.

```python
import streamlit as st

# Text files

text_contents = '''
Foo, Bar
123, 456
789, 000
'''

# Different ways to use the API

st.download_button('Download CSV', text_contents, 'text/csv')
st.download_button('Download CSV', text_contents)  # Defaults to 'text/plain'

with open('myfile.csv') as f:
   st.download_button('Download CSV', f)  # Defaults to 'text/plain'

# ---
# Binary files

binary_contents = b'whatever'

# Different ways to use the API

st.download_button('Download file', binary_contents)  # Defaults to 'application/octet-stream'

with open('myfile.zip', 'rb') as f:
   st.download_button('Download Zip', f, file_name='archive.zip')  # Defaults to 'application/octet-stream'

# You can also grab the return value of the button,
# just like with any other button.

if st.download_button(...):
   st.write('Thanks for downloading!')
```

--------------------------------

### Initialize SQL Connection with st.cache_resource

Source: https://docs.streamlit.io/develop/tutorials/databases/mssql

Establishes a connection to an SQL Server database using pyodbc and Streamlit secrets. The `@st.cache_resource` decorator ensures the connection is initialized only once, improving performance by avoiding redundant connection setups on app reruns.

```python
import streamlit as st
import pyodbc

@st.cache_resource
def init_connection():
    return pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};SERVER="
        + st.secrets["server"]
        + ";DATABASE="
        + st.secrets["database"]
        + ";UID="
        + st.secrets["username"]
        + ";PWD="
        + st.secrets["password"]
    )

conn = init_connection()
```

--------------------------------

### Streamlit Login Page App Code (Python)

Source: https://docs.streamlit.io/develop/concepts/app-testing/examples

The Streamlit application code for a login page. It uses `streamlit` and `hmac` for password checking and session state management. It defines functions for prompting login, checking passwords, and handling logout, along with the main app flow.

```Python
"""app.py"""
import streamlit as st
import hmac

st.session_state.status = st.session_state.get("status", "unverified")
st.title("My login page")


def check_password():
    if hmac.compare_digest(st.session_state.password, st.secrets.password):
        st.session_state.status = "verified"
    else:
        st.session_state.status = "incorrect"
    st.session_state.password = ""

def login_prompt():
    st.text_input("Enter password:", key="password", on_change=check_password)
    if st.session_state.status == "incorrect":
        st.warning("Incorrect password. Please try again.")

def logout():
    st.session_state.status = "unverified"

def welcome():
    st.success("Login successful.")
    st.button("Log out", on_click=logout)


if st.session_state.status != "verified":
    login_prompt()
    st.stop()
welcome()
```

--------------------------------

### Enhanced Charts with Streamlit Extras

Source: https://docs.streamlit.io/develop/api-reference/charts

Adds extra functionalities to Streamlit charts, such as annotations. This example demonstrates adding annotations to an Altair chart.

```python
import streamlit as st
# Assuming 'chart' is a pre-defined Altair chart object
# chart += get_annotations_chart(annotations=[("Mar 01, 2008", "Pretty good day for GOOG"), ("Dec 01, 2007", "Something's going wrong for GOOG & AAPL"), ("Nov 01, 2008", "Market starts again thanks to..."), ("Dec 01, 2009", "Small crash for GOOG after..."),])
# st.altair_chart(chart, use_container_width=True)
```

--------------------------------

### Configure DataFrame Column Display

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

Configures the display and editing behavior of columns in Streamlit DataFrames and data editors. This example shows how to set up a NumberColumn for price with a specific format.

```python
st.column_config.NumberColumn("Price (in USD)", min_value=0, format="$%d")
```

--------------------------------

### AppTest - Initialize from File

Source: https://docs.streamlit.io/develop/api-reference_slug=%28&slug=develop&slug=concepts&slug=configuration

Initializes a simulated Streamlit app for testing purposes from a Python file.

```APIDOC
## AppTest.from_file

### Description
`st.testing.v1.AppTest.from_file` initializes a simulated app from a file.

### Method
Python

### Endpoint
N/A

### Parameters
#### Path Parameters
*   **file_path** (str) - Required - The path to the Streamlit script file.

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("streamlit_app.py")
at.run()
```

### Response
#### Success Response (200)
*   **AppTest** - An instance of the AppTest class representing the simulated app.

#### Response Example
N/A
```

--------------------------------

### Accessing Streamlit Code Blocks with AppTest

Source: https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1

This snippet shows how to get a sequence of all st.code elements using AppTest.code. Elements are accessible by their index.

```python
from streamlit_app_test import AppTest
at = AppTest()

# Access the first code element
first_code_element = at.code[0]
```

--------------------------------

### Streamlit Extras Annotations for Charts

Source: https://docs.streamlit.io/develop/api-reference_slug=advanced-features&slug=prerelease

Adds annotations to charts within Streamlit using the streamlit_extras library. This example demonstrates adding annotations to an Altair chart.

```python
import streamlit as st
from streamlit_extras.chart_annotations import get_annotations_chart

# Assuming 'chart' is a pre-defined Altair chart object
# chart += get_annotations_chart(annotations=[
#     ("Mar 01, 2008", "Pretty good day for GOOG"),
#     ("Dec 01, 2007", "Something's going wrong for GOOG & AAPL"),
#     ("Nov 01, 2008", "Market starts again thanks to..."),
#     ("Dec 01, 2009", "Small crash for GOOG after..."),
# ])
# st.altair_chart(chart, use_container_width=True)
```

--------------------------------

### Streamlit App Version Requirement

Source: https://docs.streamlit.io/develop/tutorials/configuration-and-theming/static-fonts

Specifies the minimum required version for Streamlit to run this tutorial. Ensure you have this version or a later one installed.

```text
streamlit>=1.45.0
```

--------------------------------

### Configure Snowflake Connection Parameters in connections.toml

Source: https://docs.streamlit.io/develop/tutorials/databases/snowflake

This TOML snippet demonstrates how to configure default Snowflake connection parameters, such as account, user, private key file, role, warehouse, database, and schema, in the `.snowflake/connections.toml` file. This method is useful if you already manage connections this way.

```toml
[default]
account = "xxxxxxx-xxxxxxx"
user = "xxx"
private_key_file = "../xxx/xxx.p8"
role = "xxx"
warehouse = "xxx"
database = "xxx"
schema = "xxx"
```

--------------------------------

### Get help and display docstrings with st.help

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

The st.help function displays the docstring of an object in a nicely formatted way. This is useful for quickly accessing documentation for Streamlit functions or other Python objects.

```Python
st.help(st.write)
st.help(pd.DataFrame)
```

--------------------------------

### Streamlit App CI Workflow with GitHub Actions

Source: https://docs.streamlit.io/develop/concepts/app-testing/automate-tests

This YAML workflow configures GitHub Actions to automatically test a Streamlit application. It checks out the code, sets up Python, and uses the 'streamlit-app-action' to run smoke tests and linting. Dependencies are installed from requirements.txt.

```YAML
# .github/workflows/streamlit-app.yml
name: Streamlit app

on:
  push:
    branches: ["main"]
  pull_request:
    branches: ["main"]

permissions:
  contents: read

jobs:
  streamlit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - uses: streamlit/streamlit-app-action@v0.0.3
        with:
          app-path: streamlit_app.py

```

--------------------------------

### Streamlit: Grouping Pages into Sections with st.Page Customization

Source: https://docs.streamlit.io/develop/api-reference/navigation/st

This example illustrates how to group Streamlit pages into sections using a dictionary. It utilizes `st.Page` to provide custom titles for each page, allowing for a more organized and user-friendly navigation menu. Pages can be sourced from Python files located anywhere in the repository.

```Python
import streamlit as st

pages = {
    "Your account": [
        st.Page("create_account.py", title="Create your account"),
        st.Page("manage_account.py", title="Manage your account"),
    ],
    "Resources": [
        st.Page("learn.py", title="Learn about us"),
        st.Page("trial.py", title="Try it out"),
    ],
}

pg = st.navigation(pages)
pg.run()
```

--------------------------------

### Accessing Streamlit Errors with AppTest

Source: https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1

This snippet shows how to get a sequence of all st.error elements using AppTest.error. Elements are accessible by their index.

```python
from streamlit_app_test import AppTest
at = AppTest()

# Access the first error element
first_error = at.error[0]
```

--------------------------------

### AppTest - Initialize from String

Source: https://docs.streamlit.io/develop/api-reference_slug=%5B...slug%5D

`AppTest.from_string` initializes a simulated Streamlit app environment from a Python script provided as a string.

```APIDOC
## AppTest.from_string

### Description
`st.testing.v1.AppTest.from_string` initializes a simulated app from a string.

### Method
Python

### Endpoint
N/A (Class method)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
from streamlit.testing.v1 import AppTest

at = AppTest.from_string(app_script_as_string)
at.run()
```

### Response
#### Success Response (200)
- **AppTest object** - An instance of AppTest representing the simulated app.

#### Response Example
```json
{
  "AppTest object": "<streamlit.testing.v1.AppTest object>"
}
```
```

--------------------------------

### Streamlit Extras for Annotations

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=configuration&slug=media

Utilizes the streamlit-extras library to add annotations to charts, enhancing data interpretation. This example shows adding date-based annotations to a chart.

```python
import streamlit as st
from streamlit_extras.chart_annotations import get_annotations_chart

# Assuming 'chart' is an existing chart object (e.g., Altair)
# chart += get_annotations_chart(annotations=[
#     ("Mar 01, 2008", "Pretty good day for GOOG"),
#     ("Dec 01, 2007", "Something's going wrong for GOOG & AAPL"),
#     ("Nov 01, 2008", "Market starts again thanks to..."),
#     ("Dec 01, 2009", "Small crash for GOOG after..."),
# ],)
# st.altair_chart(chart, use_container_width=True)
```

--------------------------------

### Plost for Simple Streamlit Plotting

Source: https://docs.streamlit.io/develop/api-reference_slug=publish

Integrates the Plost library for creating simple yet effective plots within Streamlit applications. This example shows a line chart with time on the x-axis and stock values on the y-axis, with different stock names colored. Requires Plost and Streamlit.

```python
import streamlit as st
import plost
import pandas as pd

# Assuming my_dataframe is a pandas DataFrame with 'time', 'stock_value', and 'stock_name' columns
# Example DataFrame creation:
my_dataframe = pd.DataFrame({
    'time': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-01', '2023-01-02', '2023-01-03']),
    'stock_value': [100, 105, 103, 200, 210, 205],
    'stock_name': ['AAPL', 'AAPL', 'AAPL', 'GOOG', 'GOOG', 'GOOG']
})

plost.line_chart(my_dataframe, x='time', y='stock_value', color='stock_name')
```

--------------------------------

### Build Streamlit Page Dictionary by Role

Source: https://docs.streamlit.io/develop/tutorials/multipage/dynamic-navigation

Dynamically builds a dictionary of allowed pages based on the user's session state role. This enables role-based access control for different sections of the app.

```python
if st.session_state.role in ["Requester", "Admin"]:
    page_dict["Request"] = request_pages
if st.session_state.role in ["Responder", "Admin"]:
    page_dict["Respond"] = respond_pages
if st.session_state.role == "Admin":
    page_dict["Admin"] = admin_pages
```

--------------------------------

### Apply NLU with nlu.load

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

This snippet demonstrates using the NLU (Natural Language Understanding) library to load a sentiment model and predict sentiment on a given text. It's an example of integrating third-party NLP capabilities.

```Python
nlu.load("sentiment").predict("I love NLU! <3")
```

--------------------------------

### Initialize Streamlit App in Specified Directory

Source: https://docs.streamlit.io/develop/api-reference/cli/init

Initializes a new Streamlit project within a specified directory. This command generates a `requirements.txt` and `streamlit_app.py` inside the target directory and offers to run the app.

```bash
streamlit init <directory>

```

```text
CWD/
└── project/
    ├── requirements.txt
    └── streamlit_app.py

```

--------------------------------

### Define Secrets in TOML File

Source: https://docs.streamlit.io/develop/api-reference/connections

Shows the format for defining secrets within a TOML file, typically used for per-project or per-profile configurations. This example defines an `OpenAI_key`.

```python
OpenAI_key = "<YOUR_SECRET_KEY>"
```

--------------------------------

### Accessing Streamlit Chat Messages with AppTest

Source: https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1

This snippet shows how to get a sequence of all st.chat_message elements using AppTest.chat_message. Elements are accessible by their index.

```python
from streamlit_app_test import AppTest
at = AppTest()

# Access the first chat message element
first_chat_message = at.chat_message[0]
```

--------------------------------

### AppTest.from_file

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=configuration&slug=layouts

Initializes a simulated Streamlit app from a Python file for testing.

```APIDOC
## AppTest.from_file

### Description
`st.testing.v1.AppTest.from_file` initializes a simulated app from a file.

### Method
Python

### Endpoint
N/A

### Parameters
#### Path Parameters
- **file_path** (str) - Required - The path to the Streamlit script file.

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("streamlit_app.py")
at.run()
```

### Response
#### Success Response (200)
- **AppTest instance** - An instance of AppTest representing the simulated app.

#### Response Example
N/A
```

--------------------------------

### Accessing Streamlit Date Inputs with AppTest

Source: https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1

This snippet shows how to get a sequence of all st.date_input widgets using AppTest.date_input. Widgets can be accessed by index or key.

```python
from streamlit_app_test import AppTest
at = AppTest()

# Access the first date input widget
first_date_input = at.date_input[0]

# Access a date input widget by its key
date_input_by_key = at.date_input(key="my_date_key")
```

--------------------------------

### Get Image Click Coordinates

Source: https://docs.streamlit.io/develop/api-reference/data

Captures the x and y coordinates of a user's click on an image displayed in Streamlit. Requires the `streamlit-image-coordinates` component.

```python
from streamlit_image_coordinates import streamlit_image_coordinates
value = streamlit_image_coordinates("https://placekitten.com/200/300")

st.write(value)
```

--------------------------------

### Basic requirements.txt for Streamlit App

Source: https://docs.streamlit.io/deploy/concepts/dependencies

A standard `requirements.txt` file listing the core Streamlit package. This is sufficient if pandas and numpy are installed as direct dependencies of Streamlit and no other external packages are used.

```text
streamlit
pandas
numpy
```

--------------------------------

### Streamlit Extras: Annotations on Charts

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=configuration&slug=layouts

Adds custom annotations to Altair charts within Streamlit. This example demonstrates adding time-based annotations to a chart. Requires streamlit-extras and altair.

```python
from streamlit_extras.chart_annotations import get_annotations_chart
import streamlit as st
import altair as alt

# Assuming 'chart' is an existing Altair chart object
# Example: chart = alt.Chart(...).encode(...)
annotations = [
    ("Mar 01, 2008", "Pretty good day for GOOG"),
    ("Dec 01, 2007", "Something's going wrong for GOOG & AAPL"),
    ("Nov 01, 2008", "Market starts again thanks to..."),
    ("Dec 01, 2009", "Small crash for GOOG after..."),
]
chart += get_annotations_chart(annotations=annotations)
st.altair_chart(chart, use_container_width=True)
```

--------------------------------

### Streamlit Extras Metric Cards

Source: https://docs.streamlit.io/develop/api-reference/data

Provides enhanced styling for Streamlit's `st.metric` elements using the `streamlit-extras` library. This example applies custom styling to metric cards.

```python
from streamlit_extras.metric_cards import style_metric_cards

# Assuming 'col3' is a Streamlit column object
col3.metric(label="No Change", value=5000, delta=0)

style_metric_cards()
```

--------------------------------

### Visualize spaCy Entities in Streamlit

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=configuration&slug=data

Provides building blocks and visualizers for spaCy in Streamlit apps. This example visualizes named entities in a given text using specified spaCy models.

```python
import spacy_streamlit

models = ["en_core_web_sm", "en_core_web_md"]
spacy_streamlit.visualize(models, "Sundar Pichai is the CEO of Google.")
```

--------------------------------

### Display Code Block in Streamlit

Source: https://docs.streamlit.io/develop/api-reference/text

Renders a block of code, optionally with syntax highlighting, in a Streamlit app. This is useful for displaying code examples. It takes a string containing the code.

```python
st.code("a = 1234")
```

--------------------------------

### Build a Stateful Counter App with Streamlit

Source: https://docs.streamlit.io/develop/concepts/architecture/session-state

This Python code demonstrates a basic Streamlit counter app. It initializes a count, provides a button to increment it, and displays the current count. Without Session State, the count resets on each interaction.

```python
import streamlit as st

st.title('Counter Example')
count = 0

increment = st.button('Increment')
if increment:
    count += 1

st.write('Count = ', count)
```

--------------------------------

### Create Streamlit Navigation with Role-Based Pages

Source: https://docs.streamlit.io/develop/tutorials/multipage/dynamic-navigation

Creates the Streamlit navigation menu by merging account pages with role-specific pages. This is done only if the user has access to any pages.

```python
if len(page_dict) > 0:
    pg = st.navigation({"Account": account_pages} | page_dict)
```

--------------------------------

### Display a multiselect widget in Python

Source: https://docs.streamlit.io/develop/api-reference/widgets

Allows users to select multiple items from a list of choices. Returns a list of the selected items. The widget starts empty by default.

```python
choices = st.multiselect("Buy", ["milk", "apples", "potatoes"])
```

--------------------------------

### Get Help Documentation in Streamlit

Source: https://docs.streamlit.io/develop/api-reference/text

Displays the docstring of a given object (like a function or class) in a nicely formatted way within the Streamlit app. This is useful for providing inline documentation. It takes an object as an argument.

```python
st.help(st.write)
st.help(pd.DataFrame)
```

--------------------------------

### Set Streamlit App Logo

Source: https://docs.streamlit.io/develop/tutorials/multipage/dynamic-navigation

Adds a logo and an icon image to the Streamlit application. These are set once in the entrypoint file and appear across all pages.

```python
st.logo("images/horizontal_blue.png", icon_image="images/icon_blue.png")
```

--------------------------------

### Display Local Video File in Streamlit

Source: https://docs.streamlit.io/develop/api-reference/media/st

This snippet shows how to read a local video file into bytes and display it using Streamlit's video player. It requires a video file in the same directory as the script.

```python
import streamlit as st

video_file = open("myvideo.mp4", "rb")
video_bytes = video_file.read()

st.video(video_bytes)
```

--------------------------------

### Accessing Streamlit Columns with AppTest

Source: https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1

This snippet shows how to get a sequence of all columns within st.columns elements using AppTest.columns. Individual columns are accessible by their index.

```python
from streamlit_app_test import AppTest
at = AppTest()

# Access the first column element
first_column = at.columns[0]
```

--------------------------------

### Accessing Streamlit Captions with AppTest

Source: https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1

This snippet shows how to get a sequence of all st.caption elements using AppTest.caption. Elements are accessible by their index in the order they appear on the page.

```python
from streamlit_app_test import AppTest
at = AppTest()

# Access the first caption element
first_caption = at.caption[0]
```

--------------------------------

### Embed External Content with Streamlit Extras (Python)

Source: https://docs.streamlit.io/develop/api-reference/text

This example uses the 'streamlit_extras' library to embed external content, such as a link to an awesome Streamlit app. It allows for customization with labels and icons, making it easy to integrate external resources or promotions within your Streamlit application.

```python
mention(label="An awesome Streamlit App", icon="streamlit",  url="https://extras.streamlit.app")
```

--------------------------------

### Add tableauserverclient to requirements.txt

Source: https://docs.streamlit.io/develop/tutorials/databases/tableau

This command adds the `tableauserverclient` library to your project's dependencies. It's recommended to pin the version for reproducible builds. Ensure this file is included in your project's version control.

```bash
# requirements.txt
tableauserverclient==x.x.x

```

--------------------------------

### Get Image Click Coordinates

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

Captures the coordinates of user clicks on an image displayed in a Streamlit app. The `streamlit_image_coordinates` component returns the x and y coordinates of the click.

```python
from streamlit_image_coordinates import streamlit_image_coordinates
value = streamlit_image_coordinates("https://placekitten.com/200/300")

st.write(value)
```

--------------------------------

### Upload Streamlit Component Wheel to Test PyPI

Source: https://docs.streamlit.io/develop/concepts/custom-components/publish

This command uploads the built distribution files (found in the 'dist/' directory) of a Streamlit Component to the Test PyPI repository. It uses the 'twine' tool and requires authentication with a Test PyPI username ('__token__') and an API token.

```bash
python -m twine upload --repository testpypi dist/*
```

--------------------------------

### Create Settings Page Stub in Streamlit

Source: https://docs.streamlit.io/develop/tutorials/multipage/dynamic-navigation

This Python code snippet creates a basic header and a welcome message for a Streamlit settings page. It utilizes `st.session_state.role` to display the logged-in user's role, assuming it's set elsewhere. No specific dependencies beyond Streamlit are required.

```python
import streamlit as st

st.header("Settings")
st.write(f"You are logged in as {st.session_state.role}.")
```

--------------------------------

### Add PyMongo Dependency to requirements.txt

Source: https://docs.streamlit.io/develop/tutorials/databases/mongodb

This command adds the PyMongo library to your project's dependencies, ensuring it is installed when your Streamlit app is deployed. It's recommended to pin the version for consistent behavior.

```bash
# requirements.txt
pymongo==x.x.x
```

--------------------------------

### Sending and Receiving Data with Custom Components

Source: https://docs.streamlit.io/develop/concepts/custom-components/intro

This Python example illustrates how to use the declared custom component function to send data to the frontend via keyword arguments and receive data back. The return value of the component function is the data sent from the frontend.

```python
# Send data to the frontend using named arguments.
return_value = my_component(name="Blackbeard", ship="Queen Anne's Revenge")

# `my_component`'s return value is the data returned from the frontend.
st.write("Value = ", return_value)
```

--------------------------------

### Streamlit: Nesting Buttons and Widgets

Source: https://docs.streamlit.io/develop/concepts/design/buttons

Demonstrates how Streamlit handles nested buttons and other widgets within button click events. It shows that deeply nested buttons or widgets that rely on state not being saved might not execute as intended due to Streamlit's execution model.

```python
import streamlit as st

if st.button('Button 1'):
    st.write('Button 1 was clicked')
    if st.button('Button 2'):
        # This will never be executed.
        st.write('Button 2 was clicked')
```

```python
import streamlit as st

if st.button('Sign up'):
    name = st.text_input('Name')

    if name:
        # This will never be executed.
        st.success(f'Welcome {name}')
```

```python
import streamlit as st
import pandas as pd

file = st.file_uploader("Upload a file", type="csv")

if st.button('Get data'):
    df = pd.read_csv(file)
    # This display will go away with the user's next action.
    st.write(df)

if st.button('Save'):
    # This will always error.
    df.to_csv('data.csv')
```

--------------------------------

### Add Annotations to Altair Chart in Streamlit

Source: https://docs.streamlit.io/develop/api-reference/charts

Extends Altair charts with annotations for specific data points within Streamlit. This example uses `streamlit_extras` and displays the chart with annotations.

```python
chart += get_annotations_chart(annotations=[("Mar 01, 2008", "Pretty good day for GOOG"), ("Dec 01, 2007", "Something's going wrong for GOOG & AAPL"), ("Nov 01, 2008", "Market starts again thanks to..."), ("Dec 01, 2009", "Small crash for GOOG after..."),],)
st.altair_chart(chart, use_container_width=True)
```

--------------------------------

### Streamlit Login Flow with Callbacks (Python)

Source: https://docs.streamlit.io/develop/concepts/connections/authentication

This Python code implements a Streamlit login flow using callbacks for a more concise implementation. The `on_click` argument of `st.button` is used to trigger `st.login()`, simplifying the conditional logic for user authentication.

```python
import streamlit as st

if not st.user.is_logged_in:
    st.button("Log in with Google", on_click=st.login)
    st.stop()

st.button("Log out", on_click=st.logout)
st.markdown(f"Welcome! {st.user.name}")

```

--------------------------------

### Display code blocks with st.code

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

The st.code function renders a block of code, optionally with syntax highlighting. This is useful for displaying code examples or snippets within your Streamlit application.

```Python
st.code("a = 1234")
```

--------------------------------

### Combine and Display Activity Data in Streamlit

Source: https://docs.streamlit.io/develop/tutorials/elements/dataframe-row-selections

This code demonstrates how to aggregate activity data from selected rows into a dictionary, convert it to a pandas DataFrame, and then display it using Streamlit. It processes both 'activity' and 'daily_activity' columns.

```Python
activity_df = {}
for person in people:
    activity_df[df.iloc[person]["name"]] = df.iloc[person]["activity"]
activity_df = pd.DataFrame(activity_df)

daily_activity_df = {}
for person in people:
    daily_activity_df[df.iloc[person]["name"]] = df.iloc[person]["daily_activity"]
daily_activity_df = pd.DataFrame(daily_activity_df)

st.dataframe(activity_df)
st.dataframe(daily_activity_df)
```

--------------------------------

### Create and Insert Data into a SQL Server Table

Source: https://docs.streamlit.io/develop/tutorials/databases/mssql

Transact-SQL commands to switch to the 'mydb' database, create a 'mytable' with 'name' and 'pet' columns, and insert sample data.

```sql
USE mydb
CREATE TABLE mytable (name varchar(80), pet varchar(80))
INSERT INTO mytable VALUES ('Mary', 'dog'), ('John', 'cat'), ('Robert', 'bird')
GO
```

--------------------------------

### Add Annotations to Altair Chart

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=tutorials&slug=llms+&slug=build-conversational-apps

Enhances Altair charts with annotations for specific data points or time intervals. This example shows how to add text annotations to a time-series chart.

```python
import streamlit as st
import altair as alt

# Assuming 'chart' is an existing Altair chart object
# Example:
# source = pd.DataFrame({'date': pd.to_datetime(['2008-01-01', '2008-02-01', '2008-03-01']),
#                        'price': [100, 110, 105]})
# chart = alt.Chart(source).mark_line().encode(x='date', y='price')

# Function to add annotations (example implementation)
def get_annotations_chart(annotations):
    # This is a placeholder, actual implementation would create annotation layers
    return chart # Return the original chart for simplicity in this example

annotations_data = [("Mar 01, 2008", "Pretty good day for GOOG"), 
                    ("Dec 01, 2007", "Something's going wrong for GOOG & AAPL"), 
                    ("Nov 01, 2008", "Market starts again thanks to..."), 
                    ("Dec 01, 2009", "Small crash for GOOG after...")]

chart_with_annotations = get_annotations_chart(annotations=annotations_data)
st.altair_chart(chart_with_annotations, use_container_width=True)
```

--------------------------------

### Add Space in Streamlit Layouts

Source: https://docs.streamlit.io/develop/api-reference/layout

This example shows how to add vertical or horizontal space to a Streamlit layout using `st.space`. It accepts predefined size arguments like 'small'.

```python
st.space("small")
```

--------------------------------

### Visualize Activity Comparison with Streamlit Charts

Source: https://docs.streamlit.io/develop/tutorials/elements/dataframe-row-selections

This snippet illustrates how to create bar and line charts in Streamlit to compare daily and yearly activity data for selected members. It includes a conditional check to display a message when no members are selected.

```Python
if len(people) > 0:
    st.header("Daily activity comparison")
    st.bar_chart(daily_activity_df)
    st.header("Yearly activity comparison")
    st.line_chart(activity_df)
else:
    st.markdown("No members selected.")
```

--------------------------------

### Run Streamlit App from Terminal

Source: https://docs.streamlit.io/get-started/tutorials/create-an-app

Executes a Streamlit application from the command line. This command initiates the Streamlit server and opens the specified Python script in your web browser, allowing for interactive development.

```terminal
streamlit run uber_pickups.py
```

--------------------------------

### Streamlit Fragment Execution Flow Example

Source: https://docs.streamlit.io/develop/concepts/architecture/fragments

This code illustrates the execution flow of Streamlit fragments. It defines two fragments (`toggle_and_text`, `filter_and_file`) and places other widgets in the main app body. Interacting with widgets inside fragments triggers only that fragment's rerun, while interacting with widgets outside reruns the entire script.

```python
import streamlit as st

st.title("My Awesome App")

@st.fragment()
def toggle_and_text():
    cols = st.columns(2)
    cols[0].toggle("Toggle")
    cols[1].text_area("Enter text")

@st.fragment()
def filter_and_file():
    cols = st.columns(2)
    cols[0].checkbox("Filter")
    cols[1].file_uploader("Upload image")

toggle_and_text()
cols = st.columns(2)
cols[0].selectbox("Select", [1,2,3], None)
cols[1].button("Update")
filter_and_file()
```

--------------------------------

### Define Multipage App with Custom Titles and Icons using st.Page

Source: https://docs.streamlit.io/develop/concepts/multipage-apps/page-and-navigation

This example shows how to define multipage Streamlit apps with custom titles and icons for each page using `st.Page`. It also sets a consistent page title and favicon for all pages using `st.set_page_config`. The `title` and `icon` parameters in `st.Page` control the navigation menu appearance, while `st.set_page_config` affects the browser tab.

```python
import streamlit as st

create_page = st.Page("create.py", title="Create entry", icon=":material/add_circle:")
delete_page = st.Page("delete.py", title="Delete entry", icon=":material/delete:")

pg = st.navigation([create_page, delete_page])
st.set_page_config(page_title="Data manager", page_icon=":material/edit:")
pg.run()
```

--------------------------------

### Visualize Pandas DataFrames and Altair Charts in Streamlit

Source: https://docs.streamlit.io/get-started/tutorials/create-a-multipage-app

Demonstrates visualizing Pandas DataFrames and creating interactive charts using Altair within Streamlit. It fetches data from a CSV file, allows users to select countries for filtering, displays the data as a table, and generates an area chart showing agricultural production over time. Requires internet access and the `altair` library.

```python
import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError

st.set_page_config(page_title="DataFrame Demo", page_icon="📊")

st.markdown("# DataFrame Demo")
st.sidebar.header("DataFrame Demo")
st.write(
    """This demo shows how to use `st.write` to visualize Pandas DataFrames.
(Data courtesy of the [UN Data Explorer](http://data.un.org/Explorer.aspx).)"""
)


@st.cache_data
def get_UN_data():
    AWS_BUCKET_URL = "http://streamlit-demo-data.s3-us-west-2.amazonaws.com"
    df = pd.read_csv(AWS_BUCKET_URL + "/agri.csv.gz")
    return df.set_index("Region")

try:
    df = get_UN_data()
    countries = st.multiselect(
        "Choose countries", list(df.index), ["China", "United States of America"]
    )
    if not countries:
        st.error("Please select at least one country.")
    else:
        data = df.loc[countries]
        data /= 1000000.0
        st.write("### Gross Agricultural Production ($B)", data.sort_index())

        data = data.T.reset_index()
        data = pd.melt(data, id_vars=["index"]).rename(
            columns={"index": "year", "value": "Gross Agricultural Product ($B)"}
        )
        chart = (
            alt.Chart(data)
            .mark_area(opacity=0.3)
            .encode(
                x="year:T",
                y=alt.Y("Gross Agricultural Product ($B):Q", stack=None),
                color="Region:N",
            )
        )
        st.altair_chart(chart, use_container_width=True)
except URLError as e:
    st.error(
        """
        **This demo requires internet access.**
        Connection error: %s
    """ % e.reason
    )
```

--------------------------------

### Streamlit Real-time Line Chart Update

Source: https://docs.streamlit.io/get-started/tutorials/create-a-multipage-app

Demonstrates updating a Streamlit line chart in real-time with new data. It uses `st.line_chart` to create the initial chart and `chart.add_rows` to append new data points. Includes a progress bar and status text to show the update progress. Requires `streamlit`, `numpy`, and `time`.

```python
import streamlit as st
import numpy as np
import time

chart = st.line_chart(last_rows)

for i in range(1, 101):
    new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
    status_text.text("%i%% Complete" % i)
    chart.add_rows(new_rows)
    progress_bar.progress(i)
    last_rows = new_rows
    time.sleep(0.05)

progress_bar.empty()

st.button("Re-run")
```

--------------------------------

### Basic Streamlit App Structure

Source: https://docs.streamlit.io/develop/tutorials/configuration-and-theming/external-fonts-old

This Python code initializes a Streamlit application. It imports the Streamlit library, which is the only necessary step to create a basic, albeit blank, Streamlit app. This serves as a starting point for building more complex Streamlit applications.

```python
import streamlit as st
```

--------------------------------

### Running Streamlit App Tests with streamlit-app-action

Source: https://docs.streamlit.io/develop/concepts/app-testing/automate-tests

This step in the GitHub Actions workflow utilizes the 'streamlit-app-action' to execute tests. It automatically installs pytest and dependencies, runs built-in smoke tests for the app and its pages, and can optionally run other Python tests.

```YAML
- uses: streamlit/streamlit-app-action@v0.0.3
  with:
    app-path: streamlit_app.py

```

--------------------------------

### Create Bidirectional Text Input Component in Python

Source: https://docs.streamlit.io/develop/api-reference/custom-components/st.components.v2.types

This example shows how to create a custom bidirectional text input component in Streamlit. It uses HTML for the input structure and JavaScript to handle user input and update component state. The Python code wraps the component for easier use and demonstrates programmatic updates via Session State.

```python
import streamlit as st

HTML = """
    <label style='padding-right: 1em;' for='txt'>Enter text</label>
    <input id='txt' type='text' />
"""

JS = """
    export default function(component) {
        const { setStateValue, parentElement, data } = component;

        const label = parentElement.querySelector('label');
        label.innerText = data.label;

        const input = parentElement.querySelector('input');
        if (input.value !== data.value) {
            input.value = data.value ?? '';
        };

        input.onkeydown = (e) => {
            if (e.key === 'Enter') {
                setStateValue('value', e.target.value);
            }
        };

        input.onblur = (e) => {
            setStateValue('value', e.target.value);
        };
    }
"

my_component = st.components.v2.component(
    "my_text_input",
    html=HTML,
    js=JS,
)

def my_component_wrapper(
    label, *, default="", key=None, on_change=lambda: None
):
    component_state = st.session_state.get(key, {})
    value = component_state.get("value", default)
    data = {"label": label, "value": value}
    result = my_component(
        data=data,
        default={"value": value},
        key=key,
        on_value_change=on_change,
    )
    return result

st.title("My custom component")

if st.button("Hello World"):
    st.session_state["my_text_input_instance"]["value"] = "Hello World"
if st.button("Clear text"):
    st.session_state["my_text_input_instance"]["value"] = ""
result = my_component_wrapper(
    "Enter something",
    default="I love Streamlit!",
    key="my_text_input_instance",
)

st.write("Result:", result)
st.write("Session state:", st.session_state)
```

--------------------------------

### Streamlit Option Menu component in Python

Source: https://docs.streamlit.io/develop/api-reference/widgets

A third-party component for creating a single-item selection menu. It supports icons and default selections, useful for navigation or primary choices. Requires installation of `streamlit-option-menu`.

```python
from streamlit_option_menu import option_menu

option_menu("Main Menu", ["Home", 'Settings'],
  icons=['house', 'gear'], menu_icon="cast", default_index=1)
```

--------------------------------

### Setting Up Test Environment in GitHub Actions

Source: https://docs.streamlit.io/develop/concepts/app-testing/automate-tests

Configures the job environment for the GitHub Actions workflow. It specifies running on the latest Ubuntu image and includes steps to check out the repository code and set up Python version 3.11.

```YAML
jobs:
  streamlit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

```

--------------------------------

### Enable Serializable Session State with TOML Configuration

Source: https://docs.streamlit.io/develop/concepts/architecture/session-state

This snippet shows how to enable the `enforceSerializableSessionState` option in Streamlit by creating a TOML configuration file. This ensures that only pickle-serializable objects can be stored in Streamlit's session state.

```toml
# .streamlit/config.toml
[runner]
enforceSerializableSessionState = true
```

--------------------------------

### Streamlit file watcher recognizing new config.toml

Source: https://docs.streamlit.io/develop/quick-reference/release-notes

Details a bug fix where the Streamlit file watcher now recognizes a `.streamlit/config.toml` file created after the app has started, eliminating the need for a server restart.

```toml
# .streamlit/config.toml
# Example configuration file
[server]
headless = true
```

--------------------------------

### Insert Chat Message Container with Streamlit

Source: https://docs.streamlit.io/develop/api-reference/chat

This example shows how to use `st.chat_message` to create a container for displaying chat messages, such as user or AI responses. It can also contain other Streamlit elements like charts.

```python
import numpy as np
with st.chat_message("user"):
    st.write("Hello 👋")
    st.line_chart(np.random.randn(30, 3))
```

--------------------------------

### Connect to SQL Server Locally using sqlcmd

Source: https://docs.streamlit.io/develop/tutorials/databases/mssql

Command to connect to a local SQL Server instance using the sqlcmd utility. Requires specifying the server, username, and password.

```bash
sqlcmd -S localhost -U SA -P '<YourPassword>'
```

--------------------------------

### Configure Streamlit chart diverging colors

Source: https://docs.streamlit.io/develop/quick-reference/release-notes

Example of how to configure default diverging colors for Plotly, Altair, and Vega-Lite charts using `theme.chartDivergingColors`. This allows for consistent and customizable chart aesthetics.

```python
# Example usage within a Streamlit app configuration
# This is a conceptual example as the actual implementation would be in a config file or code.
# theme.chartDivergingColors = ['red', 'white', 'blue']
```

--------------------------------

### Query BigQuery Data in Streamlit

Source: https://docs.streamlit.io/develop/tutorials/databases/bigquery

This Python script demonstrates how to connect to Google BigQuery using service account credentials managed by Streamlit secrets. It defines a function to run SQL queries, utilizes Streamlit's `st.cache_data` for efficient data retrieval, and displays the results. The example queries the 'shakespeare' sample dataset.

```python
# streamlit_app.py

import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery

# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials=credentials)

# Perform query.
# Uses st.cache_data to only rerun when the query changes or after 10 min.
@st.cache_data(ttl=600)
def run_query(query):
    query_job = client.query(query)
    rows_raw = query_job.result()
    # Convert to list of dicts. Required for st.cache_data to hash the return value.
    rows = [dict(row) for row in rows_raw]
    return rows

rows = run_query("SELECT word FROM `bigquery-public-data.samples.shakespeare` LIMIT 10")

# Print results.
st.write("Some wise words from Shakespeare:")
for row in rows:
    st.write("✍️ " + row['word'])

```

--------------------------------

### Initialize Streamlit Session State (Python)

Source: https://docs.streamlit.io/develop/tutorials/execution-flow/start-and-stop-fragment-auto-reruns

Initializes Streamlit's session state for a data feed application. It sets up initial data, a boolean flag for streaming control, a callback function to toggle streaming, the app title, a slider for update frequency, and buttons to start/stop streaming. This ensures the app has default values and functional UI elements on load.

```Python
if "data" not in st.session_state:
    st.session_state.data = get_recent_data(datetime.now() - timedelta(seconds=60))
```

```Python
if "stream" not in st.session_state:
    st.session_state.stream = False
```

```Python
def toggle_streaming():
    st.session_state.stream = not st.session_state.stream
```

```Python
st.title("Data feed")
```

```Python
st.sidebar.slider(
    "Check for updates every: (seconds)", 0.5, 5.0, value=1.0, key="run_every"
)
```

```Python
st.sidebar.button(
    "Start streaming", disabled=st.session_state.stream, on_click=toggle_streaming
)
st.sidebar.button(
    "Stop streaming", disabled=not st.session_state.stream, on_click=toggle_streaming
)
```

```Python
if st.session_state.stream is True:
    run_every = st.session_state.run_every
else:
    run_every = None
```

--------------------------------

### Create a SQL Server Database

Source: https://docs.streamlit.io/develop/tutorials/databases/mssql

Transact-SQL command to create a new database named 'mydb'. This is a prerequisite for inserting data.

```sql
CREATE DATABASE mydb
GO
```

--------------------------------

### Streamlit Extras for Chart Annotations

Source: https://docs.streamlit.io/develop/api-reference_slug=publish

Utilizes the streamlit-extras library to add annotations to charts, enhancing data visualization. This example shows how to add date-based annotations to an Altair chart. Requires streamlit-extras and Altair.

```python
import streamlit as st
import altair as alt
import pandas as pd
from streamlit_extras.chart_annotations import get_annotations_chart

# Assuming 'chart' is an Altair chart object
# Example chart creation:
chart = alt.Chart(pd.DataFrame({'date': pd.to_datetime(['2008-01-01', '2008-06-01', '2009-01-01']), 'value': [10, 15, 12]})).mark_line().encode(x='date', y='value')

annotations = [
    ("Mar 01, 2008", "Pretty good day for GOOG"),
    ("Dec 01, 2007", "Something's going wrong for GOOG & AAPL"),
    ("Nov 01, 2008", "Market starts again thanks to..."),
    ("Dec 01, 2009", "Small crash for GOOG after..."),
]

chart += get_annotations_chart(annotations=annotations)
st.altair_chart(chart, use_container_width=True)
```

--------------------------------

### Streamlit App Configuration File Structure

Source: https://docs.streamlit.io/develop/api-reference/configuration

Illustrates the typical file structure for a Streamlit project, showing the location of the `.streamlit/config.toml` file which holds default application settings.

```tree
your-project/
├── .streamlit/
│   └── config.toml
└── your_app.py
```

--------------------------------

### Audio/Video Enhancements in Streamlit

Source: https://docs.streamlit.io/develop/quick-reference/release-notes/2019

Improves audio and video handling capabilities, including loading from URLs, embedding YouTube videos, and setting start positions. These enhancements provide more flexibility in media integration.

```python
st.video(url='https://www.youtube.com/watch?v=dQw4w9WgXcQ', start_time=30)
```

```python
st.audio(data='audio.mp3')
```

--------------------------------

### Initialize Streamlit App and Import Libraries

Source: https://docs.streamlit.io/develop/tutorials/execution-flow/trigger-a-full-script-rerun-from-a-fragment

This snippet shows how to create a Streamlit app file (`app.py`) and import necessary Python libraries such as streamlit, pandas, numpy, datetime, string, and time. These libraries are used for data manipulation, numerical operations, date handling, and adding delays.

```Python
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import string
import time
```

--------------------------------

### Add Supabase Connection to requirements.txt (Terminal)

Source: https://docs.streamlit.io/develop/tutorials/databases/supabase

Command to add the `st-supabase-connection` library to your `requirements.txt` file. Pinning the version is recommended for reproducible builds.

```bash
# requirements.txt
st-supabase-connection==x.x.x
```

--------------------------------

### Demonstrate Concurrency Issues with st.cache_resource in Python

Source: https://docs.streamlit.io/develop/concepts/architecture/caching

This example shows how st.cache_resource can cause concurrency issues when multiple users access and mutate a cached list simultaneously. Each user's modification affects the single cached object, leading to inconsistent results across sessions.

```python
import streamlit as st

@st.cache_resource
def create_list():
    l = [1, 2, 3]
    return l

l = create_list()
first_list_value = l[0]
l[0] = first_list_value + 1

st.write("l[0] is:", l[0])
```

--------------------------------

### Streamlit Chat message component in Python

Source: https://docs.streamlit.io/develop/api-reference/widgets

A third-party Streamlit component for creating a chatbot UI. It allows displaying chat messages with options for user or bot alignment. Requires installation of the `streamlit-chat` library.

```python
from streamlit_chat import message

message("My message")
message("Hello bot!", is_user=True)  # align's the message to the right
```

--------------------------------

### Import Libraries for Streamlit App (Python)

Source: https://docs.streamlit.io/develop/tutorials/elements/dataframe-row-selections

This Python code snippet imports essential libraries for building a Streamlit application. It includes `numpy` for numerical operations, `pandas` for data manipulation, `streamlit` for creating the web app interface, and `Faker` for generating synthetic data.

```python
import numpy as np
import pandas as pd
import streamlit as st

from faker import Faker
```

--------------------------------

### Process and Display Assistant Response with Feedback in Streamlit

Source: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/chat-response-feedback

Processes a user prompt using `chat_stream`, displays the assistant's response using `st.write_stream`, adds a feedback widget, and appends the response to the chat history. This is crucial for interactive AI applications.

```Python
with st.chat_message("assistant"):
    response = st.write_stream(chat_stream(prompt))
    st.feedback(
        "thumbs",
        key=f"feedback_{len(st.session_state.history)}",
        on_change=save_feedback,
        args=[len(st.session_state.history)],
    )
st.session_state.history.append({"role": "assistant", "content": response})
```

--------------------------------

### Read and Update Streamlit Session State

Source: https://docs.streamlit.io/develop/concepts/architecture/session-state

This Python code illustrates how to read and update values stored in Streamlit's Session State. It first initializes a session state variable and then shows how to display its value and modify it using both dictionary-like and attribute-based access.

```python
import streamlit as st

if 'key' not in st.session_state:
    st.session_state['key'] = 'value'

# Reads
st.write(st.session_state.key)

# Outputs: value

# Updates
st.session_state.key = 'value2'     # Attribute API
st.session_state['key'] = 'value2'  # Dictionary like API
```

--------------------------------

### Streamlit DataFrame Visualization with Altair

Source: https://docs.streamlit.io/get-started/tutorials/create-a-multipage-app

Shows how to display and visualize Pandas DataFrames in Streamlit using `st.write` and Altair charts. It fetches data from a URL, allows country selection, and renders both a table and an area chart of agricultural production. Requires `streamlit`, `pandas`, `altair`, and `urllib.error`.

```python
import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError

@st.cache_data
def get_UN_data():
    AWS_BUCKET_URL = "http://streamlit-demo-data.s3-us-west-2.amazonaws.com"
    df = pd.read_csv(AWS_BUCKET_URL + "/agri.csv.gz")
    return df.set_index("Region")

try:
    df = get_UN_data()
    countries = st.multiselect(
        "Choose countries", list(df.index), ["China", "United States of America"]
    )
    if not countries:
        st.error("Please select at least one country.")
    else:
        data = df.loc[countries]
        data /= 1000000.0
        st.write("### Gross Agricultural Production ($B)", data.sort_index())

        data = data.T.reset_index()
        data = pd.melt(data, id_vars=["index"]).rename(
            columns={"index": "year", "value": "Gross Agricultural Product ($B)"}
        )
        chart = (
            alt.Chart(data)
            .mark_area(opacity=0.3)
            .encode(
                x="year:T",
                y=alt.Y("Gross Agricultural Product ($B):Q", stack=None),
                color="Region:N",
            )
        )
        st.altair_chart(chart, use_container_width=True)
except URLError as e:
    st.error(
        """
        **This demo requires internet access.**

        Connection error: %s
        """ % e.reason
    )
```

--------------------------------

### Display Audio Player in Streamlit

Source: https://docs.streamlit.io/develop/api-reference/media

Provides examples for embedding an audio player in a Streamlit app, supporting inputs like NumPy arrays, audio bytes, file objects, and URLs. Allows users to play audio directly within the app.

```python
st.audio(numpy_array)
st.audio(audio_bytes)
st.audio(file)
st.audio("https://example.com/myaudio.mp3", format="audio/mp3")
```

--------------------------------

### Configure Time Column in Streamlit Data Editor

Source: https://docs.streamlit.io/develop/api-reference/data/st.column_config/st.column_config

Demonstrates how to use st.column_config.TimeColumn to configure a time column within st.data_editor. This example sets a label, minimum and maximum allowed times, a custom time format, and a step interval for user input.

```python
from datetime import time
import pandas as pd
import streamlit as st

data_df = pd.DataFrame(
    {
        "appointment": [
            time(12, 30),
            time(18, 0),
            time(9, 10),
            time(16, 25),
        ]
    }
)

st.data_editor(
    data_df,
    column_config={
        "appointment": st.column_config.TimeColumn(
            "Appointment",
            min_value=time(8, 0, 0),
            max_value=time(19, 0, 0),
            format="hh:mm a",
            step=60,
        ),
    },
    hide_index=True,
)
```

--------------------------------

### Stream Data with st.write_stream in Python

Source: https://docs.streamlit.io/develop/api-reference/write-magic/st

Demonstrates how to use st.write_stream to display streamed data, including text and a Pandas DataFrame, from a Python generator function. This example shows the basic usage and integration with a Streamlit button.

```python
import time
import numpy as np
import pandas as pd
import streamlit as st

_LOREM_IPSUM = """
Lorem ipsum dolor sit amet, **consectetur adipiscing** elit, sed do eiusmod tempor
incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis
nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
"""


def stream_data():
    for word in _LOREM_IPSUM.split(" "):
        yield word + " "
        time.sleep(0.02)

    yield pd.DataFrame(
        np.random.randn(5, 10),
        columns=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
    )

    for word in _LOREM_IPSUM.split(" "):
        yield word + " "
        time.sleep(0.02)


if st.button("Stream data"):
    st.write_stream(stream_data)
```

--------------------------------

### Declaring a Streamlit Custom Component in Python

Source: https://docs.streamlit.io/develop/concepts/custom-components/intro

This Python code snippet shows the basic setup for declaring a custom component using `streamlit.components.v1.declare_component`. It specifies the component's name and the URL of its development server.

```python
import streamlit.components.v1 as components

my_component = components.declare_component(
  "my_component",
  url="http://localhost:3001"
)
```

--------------------------------

### Build Docker Image for Streamlit App

Source: https://docs.streamlit.io/deploy/tutorials/kubernetes

Builds a Docker image for the Streamlit application. Ensure the Dockerfile and run.sh are in the same directory. Replace `<GCP_PROJECT_ID>` with your Google Cloud project ID.

```docker
docker build --platform linux/amd64 -t gcr.io/<GCP_PROJECT_ID>/k8s-streamlit:test .
```

--------------------------------

### Set Server Port and Cookie Secret via Environment Variables (Terminal)

Source: https://docs.streamlit.io/develop/concepts/configuration/options

This example shows how to configure Streamlit's server port and cookie secret using environment variables in a terminal. These variables override settings in configuration files.

```bash
export STREAMLIT_SERVER_PORT=80
export STREAMLIT_SERVER_COOKIE_SECRET=dontforgottochangeme

```

--------------------------------

### Use Streamlit Component with a DataFrame

Source: https://docs.streamlit.io/develop/concepts/custom-components

This example shows how to use the imported 'AgGrid' component by passing a pandas DataFrame to it. The component will render an interactive data grid within your Streamlit application, replacing the default DataFrame display.

```python
AgGrid(my_dataframe)
```

--------------------------------

### AppTest - Running and Assertions

Source: https://docs.streamlit.io/develop/api-reference_slug=advanced-features&slug=prerelease

Demonstrates how to run the simulated Streamlit app and perform assertions on its state and elements.

```APIDOC
## POST /api/testing/apptest/{app_test_id}/run

### Description
Runs the simulated Streamlit app and updates its state.

### Method
POST

### Endpoint
`/api/testing/apptest/{app_test_id}/run`

### Parameters
#### Path Parameters
- **app_test_id** (string) - Required - The identifier of the AppTest instance.

### Request Example
`POST /api/testing/apptest/test_12345/run`

### Response
#### Success Response (200)
Indicates that the app run was successful.

#### Response Example
```json
{
  "message": "App run completed successfully."
}
```

## POST /api/testing/apptest/{app_test_id}/assert_exception

### Description
Asserts that an exception occurred during the app run.

### Method
POST

### Endpoint
`/api/testing/apptest/{app_test_id}/assert_exception`

### Parameters
#### Path Parameters
- **app_test_id** (string) - Required - The identifier of the AppTest instance.

### Request Example
`POST /api/testing/apptest/test_12345/assert_exception`

### Response
#### Success Response (200)
Indicates that an exception was caught.

#### Response Example
```json
{
  "message": "Exception caught as expected."
}
```
```

--------------------------------

### Control Logic with Button Keys and Session State

Source: https://docs.streamlit.io/develop/concepts/design/buttons

This pattern uses a button's key to store its state in `st.session_state`. Logic can then be conditioned on this state before the button widget itself is rendered. The `.get()` method is used to safely access session state keys, returning `False` if the key doesn't exist, which is useful on the initial script run.

```python
import streamlit as st

# Use the get method since the keys won't be in session_state
# on the first script run
if st.session_state.get('clear'):
    st.session_state['name'] = ''
if st.session_state.get('streamlit'):
    st.session_state['name'] = 'Streamlit'

st.text_input('Name', key='name')

st.button('Clear name', key='clear')
st.button('Streamlit!', key='streamlit')
```

--------------------------------

### Run Streamlit Application from Terminal

Source: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/llm-quickstart

Command to execute the Streamlit application. This command should be run in the terminal from the directory containing the `streamlit_app.py` file.

```bash
streamlit run streamlit_app.py
```

--------------------------------

### Create Streamlit Plotting Demo Page

Source: https://docs.streamlit.io/get-started/tutorials/create-a-multipage-app

This Python script creates a 'Plotting Demo' page for a Streamlit application. It includes a line chart that dynamically updates with random data, demonstrating animation and progress indication. The script utilizes `streamlit`, `numpy`, and `time` libraries.

```python
import streamlit as st
import time
import numpy as np

st.set_page_config(page_title="Plotting Demo", page_icon="📈")

st.markdown("# Plotting Demo")
st.sidebar.header("Plotting Demo")
st.write(
    """This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!"""
)

progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
last_rows = np.random.randn(1, 1)
chart = st.line_chart(last_rows)

for i in range(1, 101):
    new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
    status_text.text("%i%% Complete" % i)
    chart.add_rows(new_rows)
    progress_bar.progress(i)
    last_rows = new_rows
    time.sleep(0.05)

progress_bar.empty()

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain

```

--------------------------------

### Handle Uninitialized Session State in Streamlit

Source: https://docs.streamlit.io/develop/concepts/architecture/session-state

This Python code snippet demonstrates the behavior when trying to access an uninitialized variable in Streamlit's Session State. Accessing a non-existent key will result in an exception being thrown.

```python
import streamlit as st

st.write(st.session_state['value'])

# Throws an exception!
```

--------------------------------

### Connect to Data Sources with Streamlit Connections

Source: https://docs.streamlit.io/develop/quick-reference/cheat-sheet

Explains how to establish connections to various data sources like SQL databases and Snowflake using Streamlit's `st.connection` API. It also shows how to create custom connection types.

```python
st.connection("pets_db", type="sql")
conn = st.connection("sql")
conn = st.connection("snowflake")

class MyConnection(BaseConnection[myconn.MyConnection]):
    def _connect(self, **kwargs) -> MyConnection:
        return myconn.connect(**self._secrets, **kwargs)
    def query(self, query):
        return self._instance.query(query)
```

--------------------------------

### Upload Docker Image to Google Container Registry

Source: https://docs.streamlit.io/deploy/tutorials/kubernetes

Pushes the built Docker image to Google Container Registry (GCR). This command requires authentication with Google Cloud. Replace `<GCP_PROJECT_ID>` with your project ID.

```bash
gcloud auth configure-docker
docker push gcr.io/<GCP_PROJECT_ID>/k8s-streamlit:test
```

--------------------------------

### Streamlit App to Query TigerGraph

Source: https://docs.streamlit.io/develop/tutorials/databases/tigergraph

A Streamlit application that connects to TigerGraph using pyTigerGraph and Streamlit secrets. It fetches data by running an installed query and displays the results, utilizing Streamlit's caching to optimize performance.

```python
# streamlit_app.py

import streamlit as st
import pyTigerGraph as tg

# Initialize connection.
conn = tg.TigerGraphConnection(**st.secrets["tigergraph"])
conn.apiToken = conn.getToken(conn.createSecret())

# Pull data from the graph by running the "mostDirectInfections" query.
# Uses st.cache_data to only rerun when the query changes or after 10 min.
@st.cache_data(ttl=600)
def get_data():
    most_infections = conn.runInstalledQuery("mostDirectInfections")[0]["Answer"][0]
    return most_infections["v_id"], most_infections["attributes"]

items = get_data()

# Print results.
st.title(f"Patient {items[0]} has the most direct infections")
for key, val in items[1].items():
    st.write(f"Patient {items[0]}'s {key} is {val}.")
```

--------------------------------

### Manipulate Query Parameters in Python

Source: https://docs.streamlit.io/develop/api-reference/caching-and-state

Streamlit enables direct manipulation of URL query parameters through st.query_params. You can get, set, or clear these parameters, allowing for deep linking and state synchronization with the URL.

```python
st.query_params[key] = value
st.query_params.clear()
```

--------------------------------

### AppTest - Initialize from Function

Source: https://docs.streamlit.io/develop/api-reference_slug=%5B...slug%5D

`AppTest.from_function` initializes a simulated Streamlit app environment from a Python callable function.

```APIDOC
## AppTest.from_function

### Description
`st.testing.v1.AppTest.from_function` initializes a simulated app from a function.

### Method
Python

### Endpoint
N/A (Class method)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
from streamlit.testing.v1 import AppTest

at = AppTest.from_function(app_script_as_callable)
at.run()
```

### Response
#### Success Response (200)
- **AppTest object** - An instance of AppTest representing the simulated app.

#### Response Example
```json
{
  "AppTest object": "<streamlit.testing.v1.AppTest object>"
}
```
```

--------------------------------

### Streamlit Expandable and Popover Containers

Source: https://docs.streamlit.io/develop/quick-reference/cheat-sheet

Shows how to create expandable sections using `st.expander` and popover elements using `st.popover`. These allow users to reveal or hide content, improving UI organization. The `with` notation is also demonstrated.

```python
expand = st.expander("My label", icon=":material/info:")
expand.write("Inside the expander.")
pop = st.popover("Button label")
pop.checkbox("Show all")

# You can also use "with" notation:
with expand:
    st.radio("Select one:", [1, 2])
```

--------------------------------

### Copy Local Application Code to Docker

Source: https://docs.streamlit.io/deploy/tutorials/docker

Copies all files from the current directory on the host machine into the Docker container's working directory. This is an alternative to cloning a remote repository, suitable when the Dockerfile and application code are in the same location.

```dockerfile
COPY . .
```

--------------------------------

### Configure Streamlit SSL/TLS Certificates (TOML)

Source: https://docs.streamlit.io/develop/concepts/configuration/https-support

This snippet shows how to configure Streamlit to use SSL/TLS by specifying the paths to the certificate chain file and the private key file in the `.streamlit/config.toml` file. Both files must be present when the Streamlit app starts. This configuration is not suitable for Streamlit Community Cloud.

```toml
# .streamlit/config.toml

[server]
sslCertFile = '/path/to/certchain.pem'
sslKeyFile = '/path/to/private.key'
```

--------------------------------

### st.video

Source: https://docs.streamlit.io/develop/api-reference/media/st

Displays a video player in a Streamlit application. It supports various data sources for the video, including URLs, local files, and raw data. You can also customize playback options such as start time, end time, looping, autoplay, muting, and player width.

```APIDOC
## st.video

### Description
Displays a video player.

### Method
Streamlit Component Function

### Endpoint
N/A (Client-side component)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **data** (str, Path, bytes, io.BytesIO, numpy.ndarray, or file) - Required - The video to play. Can be a URL, local file path, or raw video data.
- **format** (str) - Optional - The MIME type for the video file. Defaults to "video/mp4".
- **start_time** (int, float, timedelta, str, or None) - Optional - The time from which the element should start playing. Defaults to `None` (beginning).
- **subtitles** (str, bytes, Path, io.BytesIO, or dict) - Optional - Subtitle data for the video. Defaults to `None` (no subtitles). Supports file paths, raw content, or a dictionary for multiple tracks.
- **end_time** (int, float, timedelta, str, or None) - Optional - The time at which the element should stop playing. Defaults to `None` (end of video).
- **loop** (bool) - Optional - Whether the video should loop playback. Defaults to `False`.
- **autoplay** (bool) - Optional - Whether the video should start playing automatically. Defaults to `False`.
- **muted** (bool) - Optional - Whether the video should play with the audio silenced. Defaults to `False`.
- **width** ("stretch" or int) - Optional - The width of the video player element. Defaults to `"stretch"`.

### Request Example
```python
import streamlit as st

# Example with a URL
st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

# Example with a local file and custom start time
with open("my_video.mp4", "rb") as file_handle:
    st.video(file_handle.read(), format="video/mp4", start_time=10)

# Example with subtitles
st.video("local_video.mp4", subtitles="subtitles.vtt")

# Example with multiple subtitle tracks
st.video("local_video.mp4", subtitles={"English": "en.vtt", "French": "fr.srt"})
```

### Response
#### Success Response (200)
N/A (Streamlit component renders directly)

#### Response Example
N/A
```

--------------------------------

### Streamlit Selectbox Example

Source: https://docs.streamlit.io/develop/concepts/design/custom-classes

Demonstrates how st.selectbox stores options and selections in Streamlit's Session State. The selected value is returned by the widget function, but it's important to note that the returned value comes from an Iterable saved in Session State during a previous execution, not the current one.

```python
import streamlit as st

number = st.selectbox("Pick a number, any number", options=[1, 2, 3])
# number == whatever value the user has selected from the UI.
```

--------------------------------

### Get User Info as Dictionary in Streamlit

Source: https://docs.streamlit.io/develop/api-reference/user/st

The `st.user.to_dict()` method returns the current user's information as a dictionary. This method is primarily for internal use, as `st.user` objects inherit from `dict` by default and can usually be accessed directly.

```python
import streamlit as st

user_info_dict = st.user.to_dict()
# user_info_dict is a dictionary of the current user's information
```

--------------------------------

### Use Callbacks for Button Actions

Source: https://docs.streamlit.io/develop/concepts/design/buttons

This method utilizes the `on_click` parameter of `st.button` to directly associate a callback function with the button's press event. Arguments can be passed to the callback function using the `args` parameter. This approach simplifies state management by directly triggering functions that modify session state.

```python
import streamlit as st

st.text_input('Name', key='name')

def set_name(name):
    st.session_state.name = name

st.button('Clear name', on_click=set_name, args=[''])
st.button('Streamlit!', on_click=set_name, args=['Streamlit'])
```

--------------------------------

### Run and Interact with AppTest

Source: https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1

Demonstrates running the simulated Streamlit app and interacting with its elements. After initialization, `AppTest.run()` executes the script. Widget values can be updated, and `AppTest.run()` must be called again to reflect changes. Page switching is also handled explicitly.

```python
app = st.testing.v1.AppTest.from_file("my_app.py")
app.run()

# Update a widget value (e.g., a text input)
app.text_input[0].value = "New Value"
app.run() # Re-run the app to see the change

# Switch to another page
app.switch_page("page2.py")
app.run()
```

--------------------------------

### Place Streamlit Fragment in Sidebar

Source: https://docs.streamlit.io/develop/concepts/architecture/fragments

This example shows how to render a Streamlit fragment within a specific container, such as the sidebar. By calling the fragment function inside a `with` statement for a container (e.g., `st.sidebar`), you can control its placement while maintaining the fragment's isolated rerun behavior.

```python
import streamlit as st

@st.fragment
def fragment_function():
    if st.button("Hi!"):
        st.write("Hi back!")

with st.sidebar:
    fragment_function()
```

--------------------------------

### Streamlit Login Flow for Multiple Providers (Python)

Source: https://docs.streamlit.io/develop/concepts/connections/authentication

This Python code demonstrates how to handle login flows for multiple OIDC providers in Streamlit. It presents separate login buttons for each configured provider, allowing users to choose their authentication method. The `st.login()` function is called with the provider name as an argument.

```python
import streamlit as st

if not st.user.is_logged_in:
    if st.button("Log in with Google"):
        st.login("google")
    if st.button("Log in with Microsoft"):
        st.login("microsoft")
    st.stop()

if st.button("Log out"):
    st.logout()
st.markdown(f"Welcome! {st.user.name}")

```

--------------------------------

### Expandable Containers

Source: https://docs.streamlit.io/develop/quick-reference/cheat-sheet

Use expanders and popovers to hide or reveal content on demand.

```APIDOC
## Expandable Containers

### Description
Provide collapsible sections (expanders) and popovers to manage content visibility and user interaction.

### Method
Python

### Endpoint
N/A

### Parameters
None

### Request Example
```python
expand = st.expander("My label", icon=":material/info:")
expand.write("Inside the expander.")
pop = st.popover("Button label")
pop.checkbox("Show all")

# You can also use "with" notation:
with expand:
    st.radio("Select one:", [1, 2])
```

### Response
N/A
```

--------------------------------

### Edit Streamlit App Title in Python

Source: https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/quickstart

This snippet demonstrates how to modify the title of a Streamlit application by changing a string literal within the `st.title()` function. It's a simple text replacement for updating the app's main heading.

```python
st.title("🎈 My new Streamlit app")
```

--------------------------------

### Configure Multiple OIDC Providers in Streamlit (TOML)

Source: https://docs.streamlit.io/develop/concepts/connections/authentication

This TOML configuration sets up multiple OIDC providers for Streamlit authentication. Each provider is defined in its own dictionary under `[auth.<provider_name>]`, allowing for unique client IDs, secrets, and metadata URLs while sharing common parameters like redirect_uri and cookie_secret.

```toml
[auth]
redirect_uri = "http://localhost:8501/oauth2callback"
cookie_secret = "xxx"

[auth.google]
client_id = "xxx"
client_secret = "xxx"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

[auth.microsoft]
client_id = "xxx"
client_secret = "xxx"
server_metadata_url = "https://login.microsoftonline.com/{tenant}/v2.0/.well-known/openid-configuration"

```

--------------------------------

### Cache Database Connection with st.cache_resource

Source: https://docs.streamlit.io/develop/concepts/architecture/caching

Shows how to use `st.cache_resource` to initialize and cache a database connection object. This prevents the overhead of creating new connections for each app run or user interaction, ensuring efficient database access.

```python
@st.cache_resource
def init_connection():
    host = "hh-pgsql-public.ebi.ac.uk"
    database = "pfmegrnargs"
    user = "reader"
    password = "NWDMCE5xdipIjRrp"
    return psycopg2.connect(host=host, database=database, user=user, password=password)

conn = init_connection()
```

--------------------------------

### Get user info with st.user in Python

Source: https://docs.streamlit.io/develop/api-reference/user

The `st.user` object provides information about the currently logged-in user. It has an `is_logged_in` attribute to check authentication status and a `name` attribute to display the user's name. This is useful for personalizing the app experience.

```python
if st.user.is_logged_in:
  st.write(f"Welcome back, {st.user.name}!")
```

--------------------------------

### Import Libraries for Streamlit LLM App

Source: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/llm-quickstart

Imports the Streamlit library for building the web interface and ChatOpenAI from langchain_openai for accessing OpenAI's chat models.

```python
import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
```

--------------------------------

### Display Selected Data in Streamlit

Source: https://docs.streamlit.io/develop/tutorials/elements/dataframe-row-selections

This snippet shows how to display a header, filter a DataFrame based on row selections, and then render the filtered data in a Streamlit dataframe. It assumes 'event.selection.rows' contains the selected row indices and 'column_configuration' is pre-defined.

```Python
st.header("Selected members")
people = event.selection.rows
filtered_df = df.iloc[people]
st.dataframe(
    filtered_df,
    column_config=column_configuration,
    use_container_width=True,
)
```

--------------------------------

### Display a download button widget in Python

Source: https://docs.streamlit.io/develop/api-reference/widgets

Creates a button that allows users to download a file. Requires a file object or data to be passed as an argument. The button's label is also configurable.

```python
st.download_button("Download file", file)
```

--------------------------------

### Call the fragment function in Python

Source: https://docs.streamlit.io/develop/tutorials/execution-flow/start-and-stop-fragment-auto-reruns

This Python code snippet demonstrates how to call the previously defined fragment function `show_latest_data()` at the end of your Streamlit script. This call is essential for the fragment function to execute and display the streaming data within the application.

```Python
show_latest_data()
```

--------------------------------

### Create Bar Chart for Hourly Pickups with Streamlit and NumPy in Python

Source: https://docs.streamlit.io/get-started/tutorials/create-an-app

This Python snippet illustrates how to generate and display a bar chart representing hourly data distributions using Streamlit and NumPy. It first calculates histogram values for pickup times binned by hour using NumPy's histogram function and then visualizes these values as a bar chart using Streamlit's st.bar_chart method. This is useful for analyzing temporal patterns in data.

```python
st.subheader('Number of pickups by hour')

```

```python
hist_values = np.histogram(
    data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]

```

```python
st.bar_chart(hist_values)

```

--------------------------------

### Display Help for Pandas DataFrame using st.help

Source: https://docs.streamlit.io/develop/api-reference/utilities/st

This snippet demonstrates how to use st.help to display detailed information about the pandas DataFrame object. It's useful for quickly understanding the structure and methods of a DataFrame without needing to consult external documentation.

```python
import streamlit as st
import pandas

st.help(pandas.DataFrame)
```

--------------------------------

### Run Streamlit on a specific port

Source: https://docs.streamlit.io/knowledge-base/using-streamlit/sanity-checks

This command runs a Streamlit application on a specified port, which can help diagnose browser caching issues. By using a different port, the browser is forced to load the application fresh. Requires Streamlit and a Python script (e.g., my_app.py).

```bash
streamlit run my_app.py --server.port=9876
```

--------------------------------

### Add Tailwind CSS to Streamlit Component in Python

Source: https://docs.streamlit.io/develop/api-reference/custom-components/st.components.v2.types

This example demonstrates how to integrate Tailwind CSS into a Streamlit custom component by disabling shadow DOM isolation. It shows a simple button styled with Tailwind classes and how to mount multiple instances of the same component using different keys.

```python
import streamlit as st

with open("tailwind.js", "r") as f:
    TAILWIND_SCRIPT = f.read()

HTML = """
    <button class="bg-blue-500 hover:bg-blue-700 text-white py-1 px-3 rounded">
        Click me!
    </button>
"""
JS = (
    TAILWIND_SCRIPT
    + """
        export default function(component) {
            const { setTriggerValue, parentElement } = component;
            const button = parentElement.querySelector('button');
            button.onclick = () => {
                setTriggerValue('clicked', true);
            };
        }
    """)
my_component = st.components.v2.component(
    "my_tailwind_button",
    html=HTML,
    js=JS,
    isolate_styles=False,
)
result_1 = my_component(on_clicked_change=lambda: None, key="one")
result_1

result_2 = my_component(on_clicked_change=lambda: None, key="two")
result_2
```

--------------------------------

### Create Dynamic Navigation Menu with Streamlit

Source: https://docs.streamlit.io/develop/tutorials/multipage/dynamic-navigation

This Python code defines the main Streamlit application logic for creating a dynamic navigation menu. It handles user roles, page definitions, and navigation rendering based on the selected role. It requires Streamlit version 1.36.0 or higher.

```Python
import streamlit as st

if "role" not in st.session_state:
    st.session_state.role = None

ROLES = [None, "Requester", "Responder", "Admin"]


def login():

    st.header("Log in")
    role = st.selectbox("Choose your role", ROLES)

    if st.button("Log in"):
        st.session_state.role = role
        st.rerun()


def logout():
    st.session_state.role = None
    st.rerun()


role = st.session_state.role

logout_page = st.Page(logout, title="Log out", icon=":material/logout:")
settings = st.Page("settings.py", title="Settings", icon=":material/settings:")
request_1 = st.Page(
    "request/request_1.py",
    title="Request 1",
    icon=":material/help:",
    default=(role == "Requester"),
)
request_2 = st.Page(
    "request/request_2.py", title="Request 2", icon=":material/bug_report:"
)
respond_1 = st.Page(
    "respond/respond_1.py",
    title="Respond 1",
    icon=":material/healing:",
    default=(role == "Responder"),
)
respond_2 = st.Page(
    "respond/respond_2.py", title="Respond 2", icon=":material/handyman:"
)
admin_1 = st.Page(
    "admin/admin_1.py",
    title="Admin 1",
    icon=":material/person_add:",
    default=(role == "Admin"),
)
admin_2 = st.Page("admin/admin_2.py", title="Admin 2", icon=":material/security:")

account_pages = [logout_page, settings]
request_pages = [request_1, request_2]
respond_pages = [respond_1, respond_2]
admin_pages = [admin_1, admin_2]

st.title("Request manager")
st.logo("images/horizontal_blue.png", icon_image="images/icon_blue.png")

page_dict = {}
if st.session_state.role in ["Requester", "Admin"]:
    page_dict["Request"] = request_pages
if st.session_state.role in ["Responder", "Admin"]:
    page_dict["Respond"] = respond_pages
if st.session_state.role == "Admin":
    page_dict["Admin"] = admin_pages

if len(page_dict) > 0:
    pg = st.navigation({"Account": account_pages} | page_dict)
else:
    pg = st.navigation([st.Page(login)])

pg.run()

```

--------------------------------

### Streamlit Counter App with Callbacks and Kwargs

Source: https://docs.streamlit.io/develop/concepts/architecture/session-state

This Streamlit code snippet shows how to use keyword arguments (kwargs) with callbacks for incrementing and decrementing a counter. It defines separate functions for incrementing and decrementing, passing specific values via the 'kwargs' parameter of the button widgets.

```python
import streamlit as st

st.title('Counter Example using Callbacks with kwargs')
if 'count' not in st.session_state:
    st.session_state.count = 0

def increment_counter(increment_value=0):
    st.session_state.count += increment_value

def decrement_counter(decrement_value=0):
    st.session_state.count -= decrement_value

st.button('Increment', on_click=increment_counter,
	kwargs=dict(increment_value=5))

st.button('Decrement', on_click=decrement_counter,
	kwargs=dict(decrement_value=1))

st.write('Count = ', st.session_state.count)
```

--------------------------------

### Organize Widgets with Containers

Source: https://docs.streamlit.io/develop/concepts/design/buttons

Streamlit's `st.container` allows for flexible arrangement of widgets. You can define widgets within a container and then place that container in your script's logic in a different order than their visual appearance on the webpage. This is useful for controlling the layout and flow of your application's user interface.

```python
import streamlit as st

begin = st.container()

if st.button('Clear name'):
    st.session_state.name = ''
if st.button('Streamlit!'):
    st.session_state.name = ('Streamlit')

# The widget is second in logic, but first in display
begin.text_input('Name', key='name')
```

--------------------------------

### Build Frontend Code for Streamlit Component

Source: https://docs.streamlit.io/develop/concepts/custom-components/publish

This command compiles the frontend code (HTML, CSS, JavaScript) for a bi-directional Streamlit Component. It generates a 'build' directory containing the static assets required for the component. This step is crucial for preparing the component for distribution.

```bash
cd frontend
npm run build
```

--------------------------------

### Integrate Ace Editor Component in Streamlit

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=configuration&slug=charts

This code snippet shows how to embed the Ace code editor into a Streamlit application using the 'streamlit-ace' component. It allows users to input and display code within the Streamlit interface. The 'streamlit-ace' library must be installed.

```python
import streamlit as st
from streamlit_ace import st_ace

content = st_ace()
content
```

--------------------------------

### AppTest - Initialize from String

Source: https://docs.streamlit.io/develop/api-reference_slug=%28&slug=develop&slug=concepts&slug=configuration

Initializes a simulated Streamlit app for testing purposes from a Python script string.

```APIDOC
## AppTest.from_string

### Description
`st.testing.v1.AppTest.from_string` initializes a simulated app from a string.

### Method
Python

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
*   **script_string** (str) - Required - The Streamlit script as a string.

### Request Example
```python
from streamlit.testing.v1 import AppTest

at = AppTest.from_string(app_script_as_string)
at.run()
```

### Response
#### Success Response (200)
*   **AppTest** - An instance of the AppTest class representing the simulated app.

#### Response Example
N/A
```

--------------------------------

### Run Streamlit with Server Port Flag (Terminal)

Source: https://docs.streamlit.io/develop/concepts/configuration/options

This command demonstrates how to run a Streamlit application while specifying the server port directly as a command-line flag. This method has the highest precedence for configuration.

```bash
streamlit run your_script.py --server.port 80

```

--------------------------------

### Mounting a Bidirectional Component with ComponentRenderer

Source: https://docs.streamlit.io/develop/api-reference/custom-components/st.components.v2.types

Demonstrates how to use ComponentRenderer to mount a bidirectional component in a Streamlit application. It covers essential parameters like `key`, `data` for passing information to the frontend, `default` for initial state, and layout properties `width` and `height`. It also highlights the `callbacks` parameter for handling state updates.

```python
import streamlit as st
from streamlit.components.v2 import ComponentRenderer

# Example data to pass to the component
component_data = {"message": "Hello from Streamlit!"}

# Example default state for the component
component_default_state = {"counter": 0}

# Example callback for state change
def on_counter_change():
    st.session_state.counter = st.session_state.counter + 1

# Mount the component using ComponentRenderer
component_result = ComponentRenderer(
    key="my_component",
    data=component_data,
    default=component_default_state,
    width="stretch",
    height="content",
    callbacks={"on_counter_change": on_counter_change}
)

# Access component state and triggers from the result object
# For example, to get the current value of 'counter':
# current_counter_value = component_result.state.counter

# To trigger an action in the component (if defined):
# component_result.trigger.some_action()

st.write("Component mounted successfully!")

```

--------------------------------

### Display an audio input widget in Python

Source: https://docs.streamlit.io/develop/api-reference/widgets

Allows users to record audio using their microphone directly within the Streamlit app. Returns the recorded audio data. Requires user permission to access the microphone.

```python
speech = st.audio_input("Record a voice message")
```

--------------------------------

### Count Page Reruns with Streamlit Session State

Source: https://docs.streamlit.io/get-started/fundamentals/advanced-concepts

This example demonstrates how to use Streamlit's Session State to count the number of times a page has been rerun. It initializes a counter if it doesn't exist and increments it on each rerun, displaying the total count. This is useful for tracking user interactions or application lifecycle.

```python
import streamlit as st

if "counter" not in st.session_state:
    st.session_state.counter = 0

st.session_state.counter += 1

st.header(f"This page has run {st.session_state.counter} times.")
st.button("Run it again")
```

--------------------------------

### Initialize AppTest from File (Python)

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

Initializes `st.testing.v1.AppTest` from a Streamlit script file. This is the primary method for setting up tests for your Streamlit applications.

```python
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("streamlit_app.py")
at.run()
```

--------------------------------

### Customize Google Sheet Read Operation in Streamlit

Source: https://docs.streamlit.io/develop/tutorials/databases/private-gsheet

This Python example shows how to customize the data retrieval from a Google Sheet using `conn.read()` with optional parameters. It specifies the worksheet name, cache time-to-live (TTL), columns to use, and the number of rows to read, passing these to the underlying pandas read function.

```python
df = conn.read(
    worksheet="Sheet1",
    ttl="10m",
    usecols=[0, 1],
    nrows=3,
)

```

--------------------------------

### Handle User Input in Streamlit 'user' Stage

Source: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/validate-and-edit-chat-responses

Defines the 'user' stage logic in Streamlit, which waits for user input via `st.chat_input`. Upon receiving input, it appends the prompt to history, displays it, and prepares for the assistant's response.

```python
if st.session_state.stage == "user":
    if user_input := st.chat_input("Enter a prompt"):
        st.session_state.history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            response = st.write_stream(chat_stream())
            st.session_state.pending = response
        st.session_state.stage = "validate"
        st.rerun()
```

--------------------------------

### Command Line Interface

Source: https://docs.streamlit.io/develop/quick-reference/cheat-sheet

Common Streamlit commands for managing cache, configuration, documentation, and running applications.

```APIDOC
## Command Line Interface

### Description
Streamlit provides a set of command-line tools for various operations.

### Method
Command Line

### Endpoint
N/A

### Parameters
None

### Request Example
```bash
streamlit cache clear
streamlit config show
streamlit docs
streamlit hello
streamlit help
streamlit init
streamlit run streamlit_app.py
streamlit version
```

### Response
N/A
```

--------------------------------

### Set Sidebar Font Differently from Main App in config.toml

Source: https://docs.streamlit.io/develop/concepts/configuration/theming-customize-fonts

This example illustrates how to set a different font for the Streamlit sidebar compared to the main application content. It uses the `[theme.sidebar]` table to override the main `[theme]` settings for the font property.

```toml
[theme]
font = "serif"

[theme.sidebar]
font = "sans-serif"

```

--------------------------------

### Kubernetes Deployment Configuration with OAuth2-Proxy

Source: https://docs.streamlit.io/deploy/tutorials/kubernetes

Creates a Kubernetes configuration file including a ConfigMap for OAuth2-Proxy settings and a Deployment for the Streamlit app and its OAuth2-Proxy sidecar. It also defines a Service of type LoadBalancer. Replace placeholders like `<GOOGLE_CLIENT_ID>`, `<GOOGLE_CLIENT_SECRET>`, `<16, 24, or 32 bytes>`, and `<REDIRECT_URL>`.

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: streamlit-configmap
data:
  oauth2-proxy.cfg: |-
    http_address = "0.0.0.0:4180"
    upstreams = ["http://127.0.0.1:8501/"]
    email_domains = ["*"]
    client_id = "<GOOGLE_CLIENT_ID>"
    client_secret = "<GOOGLE_CLIENT_SECRET>"
    cookie_secret = "<16, 24, or 32 bytes>"
    redirect_url = <REDIRECT_URL>

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: streamlit-deployment
  labels:
    app: streamlit
spec:
  replicas: 1
  selector:
    matchLabels:
      app: streamlit
  template:
    metadata:
      labels:
        app: streamlit
    spec:
      containers:
        - name: oauth2-proxy
          image: quay.io/oauth2-proxy/oauth2-proxy:v7.2.0
          args: ["--config", "/etc/oauth2-proxy/oauth2-proxy.cfg"]
          ports:
            - containerPort: 4180
          livenessProbe:
            httpGet:
              path: /ping
              port: 4180
              scheme: HTTP
          readinessProbe:
            httpGet:
              path: /ping
              port: 4180
              scheme: HTTP
          volumeMounts:
            - mountPath: "/etc/oauth2-proxy"
              name: oauth2-config
        - name: streamlit
          image: gcr.io/<GCP_PROJECT_ID>/k8s-streamlit:test
          imagePullPolicy: Always
          ports:
            - containerPort: 8501
          livenessProbe:
            httpGet:
              path: /_stcore/health
              port: 8501
              scheme: HTTP
            timeoutSeconds: 1
          readinessProbe:
            httpGet:
              path: /_stcore/health
              port: 8501
              scheme: HTTP
            timeoutSeconds: 1
          resources:
            limits:
              cpu: 1
              memory: 2Gi
            requests:
              cpu: 100m
              memory: 745Mi
      volumes:
        - name: oauth2-config
          configMap:
            name: streamlit-configmap

---
apiVersion: v1
kind: Service
metadata:
  name: streamlit-service
spec:
  type: LoadBalancer
  selector:
    app: streamlit
  ports:
    - name: streamlit-port
      protocol: TCP
      port: 80
      targetPort: 4180

```

--------------------------------

### Create Streamlit Login Page Function

Source: https://docs.streamlit.io/develop/tutorials/multipage/dynamic-navigation

This Python code defines a `login` function for a Streamlit application. It includes a header, a selectbox for choosing a role from predefined options, and a login button. Clicking the button updates `st.session_state.role` and reruns the app, abstracting an authentication flow.

```python
def login():
    st.header("Log in")
    role = st.selectbox("Choose your role", ROLES)
    if st.button("Log in"):
        st.session_state.role = role
        st.rerun()
```

--------------------------------

### Watch Additional Directories for Changes with server.folderWatchList

Source: https://docs.streamlit.io/develop/quick-reference/release-notes/2025

Introduces the `server.folderWatchList` configuration option, which allows Streamlit to monitor additional directories for file changes. This enhances the hot-reloading feature by including files outside the main app directory. Configure this in `config.toml`.

```toml
[server]
folderWatchList = ["./data", "./models"]
```

--------------------------------

### Add Tags to Streamlit Apps (Community Component)

Source: https://docs.streamlit.io/develop/api-reference/text

A third-party component for adding tag input fields to Streamlit apps. It supports suggestions, maximum tags, and initial values. Requires installation of the component. Dependencies include `streamlit` and potentially others depending on the component's implementation.

```python
st_tags(label='# Enter Keywords:', text='Press enter to add more', value=['Zero', 'One', 'Two'], suggestions=['five', 'six', 'seven', 'eight', 'nine', 'three', 'eleven', 'ten', 'four'], maxtags = 4, key='1')
```

--------------------------------

### st.help

Source: https://docs.streamlit.io/develop/api-reference/utilities/st

Displays help and other information for a given object, including its name, type, value, signature, docstring, and member variables/methods.

```APIDOC
## st.help

### Description
Displays help and other information for a given object. Depending on the type of object that is passed in, this displays the object's name, type, value, signature, docstring, and member variables, methods — as well as the values/docstring of members and methods.

### Method
`st.help(obj=, *, width="stretch")`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
import streamlit as st
import pandas

st.help(pandas.DataFrame)
```

### Response
#### Success Response (200)
Information about the provided object.

#### Response Example
```json
{
  "object_name": "DataFrame",
  "object_type": "class",
  "signature": "DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)",
  "docstring": "Two-dimensional, size-mutable, potentially heterogeneous tabular data. Data structure with labeled axes (rows and columns). Similar to a spreadsheet or SQL table, or a dict of Series objects.",
  "members": [
    {
      "name": "__init__",
      "type": "method",
      "signature": "__init__(self, data=None, index=None, columns=None, dtype=None, copy=False)",
      "docstring": "Initialize a DataFrame."
    }
  ]
}
```
```

--------------------------------

### Display disabled chat input in Streamlit

Source: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/validate-and-edit-chat-responses

This Python code displays a disabled chat input widget in a Streamlit application. The placeholder text 'Accept, correct, or rewrite the answer above.' guides the user on the expected actions. This is used in the 'validate' stage to prevent further user input while they review the response.

```python
st.chat_input("Accept, correct, or rewrite the answer above.", disabled=True)
```

--------------------------------

### Update Streamlit Line Chart with New Rows

Source: https://docs.streamlit.io/develop/concepts/design/animate

This example demonstrates how to update a Streamlit line chart in-place by appending new rows of data. It uses `st.line_chart` and its `.add_rows()` method within a loop to show live data updates. Dependencies include `streamlit`, `pandas`, `numpy`, and `time`.

```python
import streamlit as st
import pandas as pd
import numpy as np
import time

df = pd.DataFrame(np.random.randn(15, 3), columns=(["A", "B", "C"]))
my_data_element = st.line_chart(df)

for tick in range(10):
    time.sleep(.5)
    add_df = pd.DataFrame(np.random.randn(1, 3), columns=(["A", "B", "C"]))
    my_data_element.add_rows(add_df)

st.button("Regenerate")
```

--------------------------------

### AppTest.from_file

Source: https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1

Create an instance of `AppTest` to simulate an app page defined within a file. This is convenient for CI workflows and testing published apps.

```APIDOC
## AppTest.from_file

### Description
Create an instance of `AppTest` to simulate an app page defined within a file. This option is most convenient for CI workflows and testing of published apps. The script must be executable on its own and so must contain all necessary imports.

### Method
POST

### Endpoint
/websites/streamlit_io/AppTest/from_file

### Parameters
#### Path Parameters
- **script_path** (str | Path) - Required - Path to a script file. The path should be absolute or relative to the file calling `.from_file`.
- **default_timeout** (float) - Optional - Default time in seconds before a script run is timed out. Can be overridden for individual `.run()` calls.

### Request Body
```json
{
  "script_path": "/path/to/your/script.py",
  "default_timeout": 5.0
}
```

### Response
#### Success Response (200)
- **AppTest Instance** (AppTest) - A simulated Streamlit app for testing. The simulated app can be executed via `.run()`.

#### Response Example
```json
{
  "message": "AppTest instance created successfully"
}
```
```

--------------------------------

### Add Containers to Fix Monthly Sales Display Height (Python)

Source: https://docs.streamlit.io/develop/tutorials/execution-flow/trigger-a-full-script-rerun-from-a-fragment

This Python code snippet illustrates using `st.container` to manage the layout for monthly sales data. It incorporates containers to define fixed heights for displaying monthly sales figures and total sales summaries, improving visual organization.

```python
def show_monthly_sales(data):
    time.sleep(1)
    selected_date = st.session_state.selected_date
    this_month = selected_date.replace(day=1)
    next_month = (selected_date.replace(day=28) + timedelta(days=4)).replace(day=1)

    st.container(height=100, border=False) ### ADD CONTAINER ###

    with st.container(height=510): ### ADD CONTAINER ###
        st.header(f"Daily sales for all products, {this_month:%B %Y}")
        monthly_sales = data[(data.index < next_month) & (data.index >= this_month)]
        st.write(monthly_sales)

    with st.container(height=510): ### ADD CONTAINER ###
        st.header(f"Total sales for all products, {this_month:%B %Y}")
        st.bar_chart(monthly_sales.sum())
```

--------------------------------

### Generate Random Member Data Function (Python)

Source: https://docs.streamlit.io/develop/tutorials/elements/dataframe-row-selections

Defines a Python function `get_profile_dataset` to generate a pandas DataFrame of random member data. It utilizes the Faker library for profile generation and NumPy for random number arrays. The function is decorated with `@st.cache_data` for performance optimization. It accepts parameters for the number of items and a random seed, returning a DataFrame with 'name', 'daily_activity', and 'activity' columns.

```Python
@st.cache_data
def get_profile_dataset(number_of_items: int = 20, seed: int = 0) -> pd.DataFrame:
    new_data = []

    fake = Faker()
    np.random.seed(seed)
    Faker.seed(seed)

    for i in range(number_of_items):
        profile = fake.profile()
        new_data.append(
            {
                "name": profile["name"],
                "daily_activity": np.random.rand(25),
                "activity": np.random.randint(2, 90, size=12),
            }
        )

    profile_df = pd.DataFrame(new_data)
    return profile_df
```

```Python
@st.cache_data
def get_profile_dataset(number_of_items: int = 20, seed: int = 0) -> pd.DataFrame:
```

```Python
new_data = []
```

```Python
fake = Faker()
random.seed(seed)
Faker.seed(seed)
```

```Python
for i in range(number_of_items):
    profile = fake.profile()
    new_data.append(
        {
            "name": profile["name"],
            "daily_activity": np.random.rand(25),
            "activity": np.random.randint(2, 90, size=12),
        }
    )
```

```Python
profile_df = pd.DataFrame(new_data)
return profile_df
```

```Python
st.dataframe(get_profile_dataset())
```

--------------------------------

### Get Filenames from Multiple File Uploads in Streamlit

Source: https://docs.streamlit.io/knowledge-base/using-streamlit/retrieve-filename-uploaded

This Python code snippet shows how to retrieve filenames when multiple files are uploaded using `st.file_uploader` with `accept_multiple_files=True`. It iterates through the list of `UploadedFile` objects and accesses the `.name` attribute for each file to display its name.

```python
import streamlit as st

uploaded_files = st.file_uploader("Upload multiple files", accept_multiple_files=True)

if uploaded_files:
   for uploaded_file in uploaded_files:
       st.write("Filename: ", uploaded_file.name)
```

--------------------------------

### Implement Stateful Counter with Streamlit Session State

Source: https://docs.streamlit.io/develop/concepts/architecture/session-state

This Python code updates the counter app to use Streamlit's Session State. By storing the 'count' in session state, its value persists across button clicks, allowing the counter to increment correctly.

```python
import streamlit as st

st.title('Counter Example')
if 'count' not in st.session_state:
    st.session_state.count = 0

increment = st.button('Increment')
if increment:
    st.session_state.count += 1

st.write('Count = ', st.session_state.count)
```

--------------------------------

### Display a file uploader widget in Python

Source: https://docs.streamlit.io/develop/api-reference/widgets

Creates a widget that allows users to upload files. Returns an `UploadedFile` object or None if no file is uploaded. Supports various file types and can be configured with specific formats.

```python
data = st.file_uploader("Upload a CSV")
```

--------------------------------

### Get Filename from Single File Upload in Streamlit

Source: https://docs.streamlit.io/knowledge-base/using-streamlit/retrieve-filename-uploaded

This Python code snippet demonstrates how to retrieve the filename of a single file uploaded via `st.file_uploader`. It checks if a file has been uploaded and then accesses the `.name` attribute of the `UploadedFile` object to display the filename.

```python
import streamlit as st

uploaded_file = st.file_uploader("Upload a file")

if uploaded_file:
   st.write("Filename: ", uploaded_file.name)
```

--------------------------------

### AppTest - Initialize from Function

Source: https://docs.streamlit.io/develop/api-reference_slug=%28&slug=develop&slug=concepts&slug=configuration

Initializes a simulated Streamlit app for testing purposes from a Python callable function.

```APIDOC
## AppTest.from_function

### Description
`st.testing.v1.AppTest.from_function` initializes a simulated app from a function.

### Method
Python

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
*   **app_func** (callable) - Required - The Streamlit app function.

### Request Example
```python
from streamlit.testing.v1 import AppTest

at = AppTest.from_function(app_script_as_callable)
at.run()
```

### Response
#### Success Response (200)
*   **AppTest** - An instance of the AppTest class representing the simulated app.

#### Response Example
N/A
```

--------------------------------

### Edit Streamlit App Title in Python

Source: https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/quickstart_slug=deploy&slug=streamlit-community-cloud&slug=get-started

This snippet demonstrates how to modify the title of a Streamlit application by changing a string literal within the `st.title()` function. It's a common task for customizing the appearance of a Streamlit app. No external dependencies are required beyond the Streamlit library itself.

```python
import streamlit as st

st.title("🎈 My new Streamlit app")
```

--------------------------------

### Build a fragment function to stream data in Python

Source: https://docs.streamlit.io/develop/tutorials/execution-flow/start-and-stop-fragment-auto-reruns

This Python code defines a fragment function `show_latest_data` that streams data. It uses the `@st.fragment(run_every=run_every)` decorator to enable streaming, retrieves the last timestamp, concatenates new data, trims the dataset to the last 100 entries, and displays it as a line chart. This function is designed to be called within a Streamlit application.

```Python
@st.fragment(run_every=run_every)
def show_latest_data():
    last_timestamp = st.session_state.data.index[-1]
    st.session_state.data = pd.concat(
        [st.session_state.data, get_recent_data(last_timestamp)]
    )
    st.session_state.data = st.session_state.data[-100:]
    st.line_chart(st.session_state.data)
```

```Python
@st.fragment(run_every=run_every)
def show_latest_data():
```

```Python
last_timestamp = st.session_state.data.index[-1]
```

```Python
st.session_state.data = pd.concat(
    [st.session_state.data, get_recent_data(last_timestamp)]
)
st.session_state.data = st.session_state.data[-100:]
```

```Python
st.line_chart(st.session_state.data)
```

--------------------------------

### Generate and Display Interactive Dataframe with Row Selection (Python)

Source: https://docs.streamlit.io/develop/tutorials/elements/dataframe-row-selections

This Python code snippet generates a Pandas DataFrame with fake user data including names and activity metrics. It then displays this DataFrame using Streamlit's `st.dataframe`, enabling multi-row selection. The selected rows are processed to create filtered views and comparison charts.

```python
import numpy as np
import pandas as pd
import streamlit as st

from faker import Faker

@st.cache_data
def get_profile_dataset(number_of_items: int = 20, seed: int = 0) -> pd.DataFrame:
    new_data = []

    fake = Faker()
    np.random.seed(seed)
    Faker.seed(seed)

    for i in range(number_of_items):
        profile = fake.profile()
        new_data.append(
            {
                "name": profile["name"],
                "daily_activity": np.random.rand(25),
                "activity": np.random.randint(2, 90, size=12),
            }
        )

    profile_df = pd.DataFrame(new_data)
    return profile_df


column_configuration = {
    "name": st.column_config.TextColumn(
        "Name", help="The name of the user", max_chars=100, width="medium"
    ),
    "activity": st.column_config.LineChartColumn(
        "Activity (1 year)",
        help="The user's activity over the last 1 year",
        width="large",
        y_min=0,
        y_max=100,
    ),
    "daily_activity": st.column_config.BarChartColumn(
        "Activity (daily)",
        help="The user's activity in the last 25 days",
        width="medium",
        y_min=0,
        y_max=1,
    ),
}

select, compare = st.tabs(["Select members", "Compare selected"])

with select:
    st.header("All members")

    df = get_profile_dataset()

    event = st.dataframe(
        df,
        column_config=column_configuration,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
    )

    st.header("Selected members")
    people = event.selection.rows
    filtered_df = df.iloc[people]
    st.dataframe(
        filtered_df,
        column_config=column_configuration,
        use_container_width=True,
    )

with compare:
    activity_df = {}
    for person in people:
        activity_df[df.iloc[person]["name"]] = df.iloc[person]["activity"]
    activity_df = pd.DataFrame(activity_df)

    daily_activity_df = {}
    for person in people:
        daily_activity_df[df.iloc[person]["name"]] = df.iloc[person]["daily_activity"]
    daily_activity_df = pd.DataFrame(daily_activity_df)

    if len(people) > 0:
        st.header("Daily activity comparison")
        st.bar_chart(daily_activity_df)
        st.header("Yearly activity comparison")
        st.line_chart(activity_df)
    else:
        st.markdown("No members selected.")
```

--------------------------------

### AppTest Initialization

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=configuration&slug=chat

Initializes `AppTest` for simulating and testing Streamlit applications.

```APIDOC
## AppTest Initialization

### Description
`st.testing.v1.AppTest` simulates a running Streamlit app for testing purposes. It provides methods to initialize the app from different sources and interact with its elements.

### Methods

#### `AppTest.from_file(filepath: str)`
Initializes a simulated app from a Python file.

```python
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("streamlit_app.py")
at.run()
```

#### `AppTest.from_string(app_script: str)`
Initializes a simulated app from a string containing the app's script.

```python
from streamlit.testing.v1 import AppTest

at = AppTest.from_string(app_script_as_string)
at.run()
```

#### `AppTest.from_function(app_function: callable)`
Initializes a simulated app from a callable function that represents the app script.

```python
from streamlit.testing.v1 import AppTest

at = AppTest.from_function(app_script_as_callable)
at.run()
```

### Common Usage
```python
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("streamlit_app.py")
at.secrets["WORD"] = "Foobar"
at.run()
assert not at.exception

at.text_input("word").input("Bazbat").run()
assert at.warning[0].value == "Try again."
```
```

--------------------------------

### Complete Interactive Streamlit App Script

Source: https://docs.streamlit.io/get-started/tutorials/create-an-app

A comprehensive Streamlit script that includes loading data, displaying a bar chart of pickups by hour, an interactive map visualization, and a slider for filtering by hour. It also features a checkbox to show/hide raw data.

```python
import streamlit as st
import pandas as pd
import numpy as np

st.title('Uber pickups in NYC')

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

data_load_state = st.text('Loading data...')
data = load_data(10000)
data_load_state.text("Done! (using st.cache_data)")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

st.subheader('Number of pickups by hour')
hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
st.bar_chart(hist_values)

# Some number in the range 0-23
hour_to_filter = st.slider('hour', 0, 23, 17)
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

st.subheader('Map of all pickups at %s:00' % hour_to_filter)
st.map(filtered_data)
```

--------------------------------

### Display Dataframe with Multi-Row Selections (Python)

Source: https://docs.streamlit.io/develop/tutorials/elements/dataframe-row-selections

Demonstrates how to display a pandas DataFrame in a Streamlit app with advanced column configurations and multi-row selection enabled. It defines custom column types (Text, LineChart, BarChart) for better data visualization and user interaction. The `on_select='rerun'` and `selection_mode='multi-row'` parameters enable dynamic updates and allow users to select multiple rows.

```Python
column_configuration = {
    "name": st.column_config.TextColumn(
        "Name", help="The name of the user", max_chars=100, width="medium"
    ),
    "activity": st.column_config.LineChartColumn(
        "Activity (1 year)",
        help="The user's activity over the last 1 year",
        width="large",
        y_min=0,
        y_max=100,
    ),
    "daily_activity": st.column_config.BarChartColumn(
        "Activity (daily)",
        help="The user's activity in the last 25 days",
        width="medium",
        y_min=0,
        y_max=1,
    ),
}
```

```Python
st.header("All members")
```

```Python
df = get_profile_dataset()
```

```Python
event = st.dataframe(
    df,
    column_config=column_configuration,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="multi-row",
)
```

--------------------------------

### Streamlit Limitation: Setting Button State via Session State

Source: https://docs.streamlit.io/develop/concepts/architecture/session-state

This Python code snippet demonstrates a limitation in Streamlit's Session State API. It shows an attempt to set the state of a `st.button` widget using session state, which will result in a `StreamlitAPIException`.

```python
import streamlit as st

if 'my_button' not in st.session_state:
    st.session_state.my_button = True
    # Streamlit will raise an Exception on trying to set the state of button

st.button('Submit', key='my_button')
```

--------------------------------

### Deploy Streamlit App and Run Pytest with JUnit XML

Source: https://docs.streamlit.io/develop/concepts/app-testing/automate-tests

This snippet configures a GitHub Actions workflow to deploy a Streamlit application and execute pytest tests. It specifies the path to the Streamlit app and provides arguments to pytest to generate a JUnit XML report, which is then processed by a separate action for display.

```yaml
-
  uses: streamlit/streamlit-app-action@v0.0.3
  with:
    app-path: streamlit_app.py
    # Add pytest-args to output junit xml
    pytest-args: -v --junit-xml=test-results.xml
-
  if: always()
  uses: pmeier/pytest-results-action@v0.6.0
  with:
    path: test-results.xml
    summary: true
    display-options: fEX
```

--------------------------------

### Generate Random Recent Data Function (Python)

Source: https://docs.streamlit.io/develop/tutorials/execution-flow/start-and-stop-fragment-auto-reruns

Defines a Python function `get_recent_data` that generates random time-series data ('A' and 'B') between a given timestamp and the current time. It ensures that no more than 60 seconds of data is returned. Dependencies include `datetime`, `numpy`, and `pandas`.

```Python
def get_recent_data(last_timestamp):
    """Generate and return data from last timestamp to now, at most 60 seconds."""
    now = datetime.now()
    if now - last_timestamp > timedelta(seconds=60):
        last_timestamp = now - timedelta(seconds=60)
    sample_time = timedelta(seconds=0.5)  # time between data points
    next_timestamp = last_timestamp + sample_time
    timestamps = np.arange(next_timestamp, now, sample_time)
    sample_values = np.random.randn(len(timestamps), 2)

    data = pd.DataFrame(sample_values, index=timestamps, columns=["A", "B"])
    return data
```

--------------------------------

### Streamlit Column Layouts

Source: https://docs.streamlit.io/develop/quick-reference/cheat-sheet

Demonstrates how to create column layouts in Streamlit for arranging elements side-by-side. It covers creating equal columns, columns with specified ratios, and aligning columns vertically.

```python
# Two equal columns:
col1, col2 = st.columns(2)
col1.write("This is column 1")
col2.write("This is column 2")

# Three different columns:
col1, col2, col3 = st.columns([3, 1, 1])
# col1 is larger.

# Bottom-aligned columns
col1, col2 = st.columns(2, vertical_alignment="bottom")

# You can also use "with" notation:
with col1:
    st.radio("Select one:", [1, 2])
```

--------------------------------

### Start 'validate' stage conditional block in Streamlit

Source: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/validate-and-edit-chat-responses

This Python code snippet initiates a conditional block to handle the 'validate' stage within a Streamlit application. It checks the current stage stored in `st.session_state.stage`. This is typically used in conjunction with other `elif` or `if` statements to manage different stages of the application's workflow.

```python
elif st.session_state.stage == "validate":
```

--------------------------------

### Set Streamlit App Title

Source: https://docs.streamlit.io/develop/tutorials/multipage/dynamic-navigation

Sets a title for the Streamlit application that will be displayed on all pages. This is typically called in the entrypoint file.

```python
st.title("Request manager")
```

--------------------------------

### Define Product Names and Average Sales

Source: https://docs.streamlit.io/develop/tutorials/execution-flow/trigger-a-full-script-rerun-from-a-fragment

This Python code segment defines a list of product names (e.g., 'Widget A', 'Widget B') and assigns a randomly generated average daily sales value to each. It uses `string.ascii_uppercase` for product naming and `numpy.random.normal` for sales value generation.

```Python
product_names = ["Widget " + letter for letter in string.ascii_uppercase]
average_daily_sales = np.random.normal(1_000, 300, len(product_names))
products = dict(zip(product_names, average_daily_sales))
```

--------------------------------

### Build a Streamlit Chat Interface with Simulated Stream (Python)

Source: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/validate-and-edit-chat-responses

This Python code implements a Streamlit chat interface. It uses a generator function 'chat_stream' to simulate responses, a 'validate' function to check response quality, and 'add_highlights' to visually indicate potential errors. The interface manages conversation state through 'st.session_state' to handle user input, assistant responses, and validation steps.

```python
import streamlit as st
import lorem
from random import randint
import time

if "stage" not in st.session_state:
    st.session_state.stage = "user"
    st.session_state.history = []
    st.session_state.pending = None
    st.session_state.validation = {}


def chat_stream():
    for i in range(randint(3, 9)):
        yield lorem.sentence() + " "
        time.sleep(0.2)


def validate(response):
    response_sentences = response.split(". ")
    response_sentences = [
        sentence.strip(". ") + "."
        for sentence in response_sentences
        if sentence.strip(". ") != ""
    ]
    validation_list = [
        True if sentence.count(" ") > 4 else False for sentence in response_sentences
    ]
    return response_sentences, validation_list


def add_highlights(response_sentences, validation_list, bg="red", text="red"):
    return [
        f":{text}[:{bg}-background[" + sentence + "]]" if not is_valid else sentence
        for sentence, is_valid in zip(response_sentences, validation_list)
    ]


for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.stage == "user":
    if user_input := st.chat_input("Enter a prompt"):
        st.session_state.history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            response = st.write_stream(chat_stream())
            st.session_state.pending = response
            st.session_state.stage = "validate"
            st.rerun()

elif st.session_state.stage == "validate":
    st.chat_input("Accept, correct, or rewrite the answer above.", disabled=True)
    response_sentences, validation_list = validate(st.session_state.pending)
    highlighted_sentences = add_highlights(response_sentences, validation_list)
    with st.chat_message("assistant"):
        st.markdown(" ".join(highlighted_sentences))
        st.divider()
        cols = st.columns(3)
        if cols[0].button(
            "Correct errors", type="primary", disabled=all(validation_list)
        ):
            st.session_state.validation = {
                "sentences": response_sentences,
                "valid": validation_list,
            }
            st.session_state.stage = "correct"
            st.rerun()
        if cols[1].button("Accept"):
            st.session_state.history.append(
                {"role": "assistant", "content": st.session_state.pending}
            )
            st.session_state.pending = None
            st.session_state.validation = {}
            st.session_state.stage = "user"
            st.rerun()
        if cols[2].button("Rewrite answer", type="tertiary"):
            st.session_state.stage = "rewrite"
            st.rerun()

elif st.session_state.stage == "correct":
    st.chat_input("Accept, correct, or rewrite the answer above.", disabled=True)
    response_sentences = st.session_state.validation["sentences"]
    validation_list = st.session_state.validation["valid"]
    highlighted_sentences = add_highlights(
        response_sentences, validation_list, "gray", "gray"
    )
    if not all(validation_list):
        focus = validation_list.index(False)
        highlighted_sentences[focus] = ":red[:red" + highlighted_sentences[focus][11:]
    else:
        focus = None
    with st.chat_message("assistant"):
        st.markdown(" ".join(highlighted_sentences))
        st.divider()
        if focus is not None:
            new_sentence = st.text_input(
                "Replacement text:", value=response_sentences[focus]
            )
            cols = st.columns(2)
            if cols[0].button(
                "Update", type="primary", disabled=len(new_sentence.strip()) < 1
            ):
                st.session_state.validation["sentences"][focus] = (
                    new_sentence.strip(". ") + "."
                )
                st.session_state.validation["valid"][focus] = True
                st.session_state.pending = " ".join(
                    st.session_state.validation["sentences"]
                )
                st.rerun()
            if cols[1].button("Remove"):
                st.session_state.validation["sentences"].pop(focus)
                st.session_state.validation["valid"].pop(focus)
                st.session_state.pending = " ".join(
                    st.session_state.validation["sentences"]
                )
                st.rerun()
        else:
            cols = st.columns(2)

```

--------------------------------

### Extract Font Face Declarations from CSS

Source: https://docs.streamlit.io/develop/tutorials/configuration-and-theming/external-fonts-old

This CSS snippet shows an example of a `@font-face` declaration, which is used to load custom fonts. It specifies the font family, style, weight, display behavior, and the URL for the font file. This is typically found within a stylesheet linked from Google Fonts.

```css
/* cyrillic-ext */
@font-face {
  font-family: "Nunito";
  font-style: italic;
  font-weight: 200 1000;
  font-display: swap;
  src: url(https://fonts.gstatic.com/s/nunito/v31/XRXX3I6Li01BKofIMNaORs7nczIHNHI.woff2)
    format("woff2");
  unicode-range:
    U+0460-052F, U+1C80-1C8A, U+20B4, U+2DE0-2DFF, U+A640-A69F, U+FE2E-FE2F;
}
```

--------------------------------

### Display Interactive DataFrames with st.dataframe()

Source: https://docs.streamlit.io/get-started/fundamentals/main-concepts

Shows how to display a NumPy array as an interactive table using st.dataframe(). This method allows for customization, such as highlighting specific elements using Pandas Styler objects.

```python
import streamlit as st
import numpy as np

dataframe = np.random.randn(10, 20)
st.dataframe(dataframe)
```

--------------------------------

### Displaying and Executing Code with st.echo

Source: https://docs.streamlit.io/develop/api-reference_slug=private-gsheet

The `st.echo` context manager displays code within the app and then executes it, useful for tutorials.

```APIDOC
## POST /st.echo

### Description
Display some code in the app, then execute it. Useful for tutorials.

### Method
POST

### Endpoint
/st.echo

### Parameters
#### Request Body
- **code_block** (string) - Required - The block of code to display and execute.

### Request Example
```json
{
  "code_block": "with st.echo():\n  st.write('This code will be printed')"
}
```

### Response
#### Success Response (200)
- **status** (string) - Indicates successful display and execution of the code.

#### Response Example
```json
{
  "status": "success"
}
```
```

--------------------------------

### Define Basic Multipage App Structure with st.navigation

Source: https://docs.streamlit.io/develop/concepts/multipage-apps/page-and-navigation

This snippet demonstrates the basic structure for a multipage Streamlit app. It initializes navigation with two pages defined by separate Python files and then runs the selected page. Ensure `page_1.py` and `page_2.py` exist in the same directory as `streamlit_app.py`.

```python
import streamlit as st

pg = st.navigation([st.Page("page_1.py"), st.Page("page_2.py")])
pg.run()
```

--------------------------------

### Global Configuration Options in config.toml

Source: https://docs.streamlit.io/develop/api-reference/configuration/config

Illustrates global configuration settings within a config.toml file. These options control Streamlit's general behavior, such as warnings for widget state duplication and direct script execution.

```toml
[global]

# By default, Streamlit displays a warning when a user sets both a widget
# default value in the function defining the widget and a widget value via
# the widget's key in `st.session_state`.
#
# If you'd like to turn off this warning, set this to True.
#
# Default: false
disableWidgetStateDuplicationWarning = false

# If True, will show a warning when you run a Streamlit-enabled script
# via "python my_script.py".
#
# Default: true
showWarningOnDirectExecution = true
```

--------------------------------

### Initialize Streamlit Session State Role

Source: https://docs.streamlit.io/develop/tutorials/multipage/dynamic-navigation

This Python code snippet initializes the 'role' key in Streamlit's session state if it doesn't already exist. It sets the initial value to `None`, typically representing an unauthenticated user. This is crucial for managing user roles and access control within the application.

```python
if "role" not in st.session_state:
    st.session_state.role = None
```

--------------------------------

### Simulate Chat Response Stream Generator in Python

Source: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/validate-and-edit-chat-responses

This Python function, `chat_stream`, simulates a streaming chat response. It uses the 'lorem' library to generate a random number of sentences (3-9) and yields each sentence followed by a space, with a small delay (0.2 seconds) between each yield to mimic real-time streaming. It has no arguments and returns a generator.

```python
def chat_stream():
    for i in range(randint(3, 9)):
        yield lorem.sentence() + " "
        time.sleep(0.2)
```

--------------------------------

### Implement Row Selection in Dataframes (Python)

Source: https://docs.streamlit.io/develop/tutorials/elements/dataframe-row-selections-old

This Python code demonstrates how to allow users to select rows from a Pandas DataFrame in Streamlit versions prior to 1.35.0. It adds a temporary 'Select' checkbox column to the DataFrame, uses `st.data_editor` for user interaction, and then filters the DataFrame to return only the selected rows. This approach prevents mutation of the original DataFrame.

```python
import streamlit as st
import numpy as np
import pandas as pd

df = pd.DataFrame(
    {
        "Animal": ["Lion", "Elephant", "Giraffe", "Monkey", "Zebra"],
        "Habitat": ["Savanna", "Forest", "Savanna", "Forest", "Savanna"],
        "Lifespan (years)": [15, 60, 25, 20, 25],
        "Average weight (kg)": [190, 5000, 800, 10, 350],
    }
)

def dataframe_with_selections(df):
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", False)

    # Get dataframe row-selections from user with st.data_editor
    edited_df = st.data_editor(
        df_with_selections,
        hide_index=True,
        column_config={"Select": st.column_config.CheckboxColumn(required=True)},
        disabled=df.columns,
    )

    # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df.Select]
    return selected_rows.drop('Select', axis=1)


selection = dataframe_with_selections(df)
st.write("Your selection:")
st.write(selection)
```

--------------------------------

### Connect to Database and Query Data with Streamlit

Source: https://docs.streamlit.io/get-started/fundamentals/advanced-concepts

This Python snippet shows how to establish a database connection using Streamlit's `st.connection` and then query data from a table. It assumes a connection named 'my_database' is configured.

```python
import streamlit as st

conn = st.connection("my_database")
df = conn.query("select * from my_table")
st.dataframe(df)
```

--------------------------------

### Configure Single OIDC Provider in Streamlit (TOML)

Source: https://docs.streamlit.io/develop/concepts/connections/authentication

This TOML configuration sets up a single OIDC provider for Streamlit authentication. It requires shared parameters like redirect_uri and cookie_secret, along with provider-specific details such as client_id, client_secret, and server_metadata_url. Ensure the redirect_uri port matches your Streamlit app's port.

```toml
[auth]
redirect_uri = "http://localhost:8501/oauth2callback"
cookie_secret = "xxx"
client_id = "xxx"
client_secret = "xxx"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

```

--------------------------------

### Define Streamlit Request Pages with Default

Source: https://docs.streamlit.io/develop/tutorials/multipage/dynamic-navigation

Defines Streamlit pages for requests, including setting a default page based on the user's role. This allows for role-specific default navigation.

```python
request_1 = st.Page(
    "request/request_1.py",
    title="Request 1",
    icon=":material/help:",
    default=(role == "Requester"),
)
request_2 = st.Page(
    "request/request_2.py", title="Request 2", icon=":material/bug_report:"
)
```

--------------------------------

### Streamlit Tabbed Layouts

Source: https://docs.streamlit.io/develop/quick-reference/cheat-sheet

Illustrates how to create tabbed interfaces in Streamlit using `st.tabs`. This allows organizing content into different sections accessible via tabs. It also shows the use of the `with` notation for adding content to tabs.

```python
# Insert containers separated into tabs:
tab1, tab2 = st.tabs(["Tab 1", "Tab2"])
tab1.write("this is tab 1")
tab2.write("this is tab 2")

# You can also use "with" notation:
with tab1:
    st.radio("Select one:", [1, 2])
```

--------------------------------

### Create Multipage App Navigation with Streamlit

Source: https://docs.streamlit.io/get-started/fundamentals/additional-features

This snippet demonstrates how to create a multipage Streamlit application. It involves defining individual pages using `st.Page` and then organizing them into a navigation structure with `st.navigation`. The entry point script connects these pages, allowing users to navigate between them seamlessly.

```Python
import streamlit as st

# Define the pages
main_page = st.Page("main_page.py", title="Main Page", icon="🎈")
page_2 = st.Page("page_2.py", title="Page 2", icon="❄️")
page_3 = st.Page("page_3.py", title="Page 3", icon="🎉")

# Set up navigation
pg = st.navigation([main_page, page_2, page_3])

# Run the selected page
pg.run()
```

```Python
import streamlit as st

# Main page content
st.markdown("# Main page 🎈")
st.sidebar.markdown("# Main page 🎈")
```

```Python
import streamlit as st

st.markdown("# Page 2 ❄️")
st.sidebar.markdown("# Page 2 ❄️")
```

```Python
import streamlit as st

st.markdown("# Page 3 🎉")
st.sidebar.markdown("# Page 3 🎉")
```

--------------------------------

### Streamlit: Testing Helper Functions in Containers

Source: https://docs.streamlit.io/develop/tutorials/execution-flow/create-a-multiple-container-fragment

Demonstrates how to call the `black_cats` and `orange_cats` helper functions within specific containers defined earlier in the app. This is for testing purposes and should be removed later.

```python
with grid[0]:
    black_cats()
with grid[1]:
    orange_cats()
```

--------------------------------

### Add Containers to Fix Daily Sales Display Height (Python)

Source: https://docs.streamlit.io/develop/tutorials/execution-flow/trigger-a-full-script-rerun-from-a-fragment

This Python code snippet demonstrates how to use `st.container` with a specified height to fix the display area for daily sales data. It includes containers for date input, best sellers, and worst sellers, ensuring a consistent layout.

```python
@st.fragment
def show_daily_sales(data):
    time.sleep(1)
    with st.container(height=100): ### ADD CONTAINER ###
        selected_date = st.date_input(
            "Pick a day ",
            value=date(2023, 1, 1),
            min_value=date(2023, 1, 1),
            max_value=date(2023, 12, 31),
            key="selected_date",
        )

    if "previous_date" not in st.session_state:
        st.session_state.previous_date = selected_date
    previous_date = st.session_state.previous_date
    st.session_state.previous_date = selected_date
    is_new_month = selected_date.replace(day=1) != previous_date.replace(day=1)
    if is_new_month:
        st.rerun()

    with st.container(height=510): ### ADD CONTAINER ###
        st.header(f"Best sellers, {selected_date:%m/%d/%y}")
        top_ten = data.loc[selected_date].sort_values(ascending=False)[0:10]
        cols = st.columns([1, 4])
        cols[0].dataframe(top_ten)
        cols[1].bar_chart(top_ten)

    with st.container(height=510): ### ADD CONTAINER ###
        st.header(f"Worst sellers, {selected_date:%m/%d/%y}")
        bottom_ten = data.loc[selected_date].sort_values()[0:10]
        cols = st.columns([1, 4])
        cols[0].dataframe(bottom_ten)
        cols[1].bar_chart(bottom_ten)
```

--------------------------------

### Define Project Dependencies for Streamlit App

Source: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/llm-quickstart

Creates a requirements.txt file listing the Python dependencies needed for the Streamlit application. This is essential for deployment to platforms like Streamlit Community Cloud.

```python
streamlit
openai
langchain
```

--------------------------------

### Define Streamlit Account Pages

Source: https://docs.streamlit.io/develop/tutorials/multipage/dynamic-navigation

Defines Streamlit pages for account-related functions like logging out and accessing settings. Each page is given a title and an icon for the navigation menu.

```python
logout_page = st.Page(logout, title="Log out", icon=":material/logout:")
settings = st.Page("settings.py", title="Settings", icon=":material/settings:")
```

--------------------------------

### Implement Top Navigation with st.navigation

Source: https://docs.streamlit.io/develop/quick-reference/release-notes/2025

Introduces top navigation for Streamlit apps. Use the `st.navigation` function with the `position='top'` argument to create a navigation menu at the top of your application. This feature enhances user experience by providing a clear navigation structure.

```python
import streamlit as st

st.navigation([
    st.Page("app.py", "Home"),
    st.Page("pages/about.py", "About"),
    st.Page("pages/contact.py", "Contact")
], position="top")
```

--------------------------------

### Connect to Supabase and Query Data in Streamlit

Source: https://docs.streamlit.io/develop/tutorials/databases/supabase

This Python code demonstrates how to initialize a connection to Supabase using Streamlit secrets and query data from a table. It utilizes Streamlit's caching decorators (`@st.cache_resource` and `@st.cache_data`) to optimize performance by caching connection objects and query results.

```python
# streamlit_app.py

import streamlit as st
from supabase import create_client, Client

# Initialize connection.
# Uses st.cache_resource to only run once.
@st.cache_resource
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_connection()

# Perform query.
# Uses st.cache_data to only rerun when the query changes or after 10 min.
@st.cache_data(ttl=600)
def run_query():
    return supabase.table("mytable").select("*").execute()

rows = run_query()

# Print results.
for row in rows.data:
    st.write(f"{row['name']} has a :{row['pet']}:")
```

--------------------------------

### Add Streamlit App Badge to GitHub README

Source: https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app

Embed a badge in your GitHub repository's README.md file to link to your deployed Streamlit app. This badge helps others discover and interact with your application. Ensure you replace the placeholder URL with your app's actual URL.

```markdown
# Share your app
Now that your app is deployed you can easily share it and collaborate on it. But first, let's take a moment and do a little joy dance for getting that app deployed! 🕺💃
Your app is now live at a fixed URL, so go wild and share it with whomever you want. Your app will inherit permissions from your GitHub repo, meaning that if your repo is private your app will be private and if your repo is public your app will be public. If you want to change that you can simply do so from the app settings menu.
You are only allowed one private app at a time. If you've deployed from a private repository, you will have to make that app public or delete it before you can deploy another app from a private repository. Only developers can change your app between public and private.
  * Make your app public or private
  * Share your public app
  * Share your private app


## Make your app public or private
If you deployed your app from a public repository, your app will be public by default. If you deployed your app from a private repository, you will need to make the app public if you want to freely share it with the community at large.
### Set privacy from your app settings
  1. Access your App settings and go to the "**Sharing** " section.
  2. Set your app's privacy under "Who can view this app." Select "**This app is public and searchable** " to make your app public. Select "**Only specific people can view this app** " to make your app private.


### Set privacy from the share button
  1. From your app at `<your-custom-subdomain>.streamlit.app`, click "**Share** " in the upper-right corner.
  2. Toggle your app between public and private by clicking "**Make this app public**."


## Share your public app
Once your app is public, just give anyone your app's URL and they view it! Streamlit Community Cloud has several convenient shortcuts for sharing your app.
### Share your app on social media
  1. From your app at `<your-custom-subdomain>.streamlit.app`, click "**Share** " in the upper-right corner.
  2. Click "**Social** " to access convenient social media share buttons.

_star_
#### Tip
Use the social media sharing buttons to post your app on our forum! We'd love to see what you make and perhaps feature your app as our app of the month. 💖
### Invite viewers by email
Whether your app is public or private, you can send an email invite to your app directly from Streamlit Community Cloud. This grants the viewer access to analytics for all your public apps and the ability to invite other viewers to your workspace. Developers and invited viewers are identified by their email in analytics instead of appearing anonymously (if they view any of your apps while signed in). Read more about viewers in App analytics.
  1. From your app at `<your-custom-subdomain>.streamlit.app`, click "**Share** " in the upper-right corner.
  2. Enter an email address and click "**Invite**."
  3. Invited users will get a direct link to your app in their inbox.


### Copy your app's URL
From your app click "**Share** " in the upper-right corner then click "**Copy link**."
### Add a badge to your GitHub repository
To help others find and play with your Streamlit app, you can add Streamlit's GitHub badge to your repo. Below is an enlarged example of what the badge looks like. Clicking on the badge takes you to—in this case—Streamlit's Roadmap.
Once you deploy your app, you can embed this badge right into your GitHub README.md by adding the following Markdown:
Markdown
```
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://<your-custom-subdomain>.streamlit.app)
```

Copy
_push_pin_
#### Note
Be sure to replace `https://<your-custom-subdomain>.streamlit.app` with the URL of your deployed app!
## Share your private app
By default an app deployed from a private repository will be private to the developers in the workspace. A private app will not be visible to anyone else unless you grant them explicit permission. You can grant permission by adding them as a developer on GitHub or by adding them as a viewer on Streamlit Community Cloud.
Once you have added someone's email address to your app's viewer list, that person will be able to sign in and view your private app. If their email is associated with a Google account, they will be able to sign in with Google OAuth. Otherwise, they will be able to sign in with single-use, emailed links. Streamlit sends an email invitation with a link to your app every time you invite someone.
_priority_high_

```

--------------------------------

### Create Interactive Widgets with st.slider()

Source: https://docs.streamlit.io/get-started/fundamentals/main-concepts

Demonstrates the use of st.slider() to create an interactive slider widget. The value selected by the user is assigned to a variable, and the script reruns to update the displayed output based on the new value.

```python
import streamlit as st
x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)
```

--------------------------------

### Live Camera Input for Streamlit (Python)

Source: https://docs.streamlit.io/develop/api-reference/widgets

Offers an alternative to `st.camera_input` that provides live webcam image feeds. The captured images can then be processed or displayed within the Streamlit application.

```python
from camera_input_live import camera_input_live
import streamlit as st

image = camera_input_live()
st.image(image)
```

--------------------------------

### Initialize AppTest from File

Source: https://docs.streamlit.io/develop/api-reference/app-testing

Initializes a simulated Streamlit app for testing directly from a Python file. This method allows setting secrets and running the app to check for exceptions.

```Python
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("streamlit_app.py")
at.secrets["WORD"] = "Foobar"
at.run()
assert not at.exception
```

--------------------------------

### AppTest.from_function

Source: https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1

Create an instance of `AppTest` to simulate an app page defined within a function. Convenient for IDE assistance.

```APIDOC
## AppTest.from_function

### Description
Create an instance of `AppTest` to simulate an app page defined within a function. This is similar to `AppTest.from_string()`, but more convenient to write with IDE assistance. The script must be executable on its own and so must contain all necessary imports.

### Method
POST

### Endpoint
/websites/streamlit_io/AppTest/from_function

### Parameters
#### Path Parameters
- **script** (Callable) - Required - A function whose body will be used as a script. Must be runnable in isolation, so it must include any necessary imports.
- **default_timeout** (float) - Optional - Default time in seconds before a script run is timed out. Can be overridden for individual `.run()` calls.
- **args** (tuple) - Optional - An optional tuple of args to pass to the script function.
- **kwargs** (dict) - Optional - An optional dict of kwargs to pass to the script function.

### Request Body
```json
{
  "script": "def my_app():\n    import streamlit as st\n    st.write('Hello from function!')",
  "default_timeout": 5.0,
  "args": [],
  "kwargs": {}
}
```

### Response
#### Success Response (200)
- **AppTest Instance** (AppTest) - A simulated Streamlit app for testing. The simulated app can be executed via `.run()`.

#### Response Example
```json
{
  "message": "AppTest instance created successfully"
}
```
```

--------------------------------

### Configure Local Streamlit Secrets for SQL Server

Source: https://docs.streamlit.io/develop/tutorials/databases/mssql

TOML format for the `.streamlit/secrets.toml` file, storing SQL Server connection details like server address, database name, username, and password.

```toml
# .streamlit/secrets.toml

server = "localhost"
database = "mydb"
username = "SA"
password = "xxx"
```

--------------------------------

### Streamlit Mapping Demo with Pydeck

Source: https://docs.streamlit.io/get-started/tutorials/create-a-multipage-app

This demo showcases the use of `st.pydeck_chart` to display geospatial data using Pydeck. It fetches data from JSON files, defines various map layers (HexagonLayer, ScatterplotLayer, TextLayer, ArcLayer), and allows users to select layers via checkboxes in the sidebar. Requires pandas and pydeck. Handles URLError for connection issues.

```python
import streamlit as st
import pandas as pd
import pydeck as pdk

from urllib.error import URLError

def mapping_demo():
    import streamlit as st
    import pandas as pd
    import pydeck as pdk

    from urllib.error import URLError

    st.markdown(f"# {list(page_names_to_funcs.keys())[2]}")
    st.write(
        """
        This demo shows how to use
[`st.pydeck_chart`](https://docs.streamlit.io/develop/api-reference/charts/st.pydeck_chart)
        to display geospatial data.
"""
    )

    @st.cache_data
    def from_data_file(filename):
        url = (
            "http://raw.githubusercontent.com/streamlit/"
            "example-data/master/hello/v1/%s" % filename
        )
        return pd.read_json(url)

    try:
        ALL_LAYERS = {
            "Bike Rentals": pdk.Layer(
                "HexagonLayer",
                data=from_data_file("bike_rental_stats.json"),
                get_position=["lon", "lat"],
                radius=200,
                elevation_scale=4,
                elevation_range=[0, 1000],
                extruded=True,
            ),
            "Bart Stop Exits": pdk.Layer(
                "ScatterplotLayer",
                data=from_data_file("bart_stop_stats.json"),
                get_position=["lon", "lat"],
                get_color=[200, 30, 0, 160],
                get_radius="[exits]",
                radius_scale=0.05,
            ),
            "Bart Stop Names": pdk.Layer(
                "TextLayer",
                data=from_data_file("bart_stop_stats.json"),
                get_position=["lon", "lat"],
                get_text="name",
                get_color=[0, 0, 0, 200],
                get_size=15,
                get_alignment_baseline="'bottom'",
            ),
            "Outbound Flow": pdk.Layer(
                "ArcLayer",
                data=from_data_file("bart_path_stats.json"),
                get_source_position=["lon", "lat"],
                get_target_position=["lon2", "lat2"],
                get_source_color=[200, 30, 0, 160],
                get_target_color=[200, 30, 0, 160],
                auto_highlight=True,
                width_scale=0.0001,
                get_width="outbound",
                width_min_pixels=3,
                width_max_pixels=30,
            ),
        }
        st.sidebar.markdown("### Map Layers")
        selected_layers = [
            layer
            for layer_name, layer in ALL_LAYERS.items()
            if st.sidebar.checkbox(layer_name, True)
        ]
        if selected_layers:
            st.pydeck_chart(
                pdk.Deck(
                    map_style="mapbox://styles/mapbox/light-v9",
                    initial_view_state={
                        "latitude": 37.76,
                        "longitude": -122.4,
                        "zoom": 11,
                        "pitch": 50,
                    },
                    layers=selected_layers,
                )
            )
        else:
            st.error("Please choose at least one layer above.")
    except URLError as e:
        st.error(
            """
            **This demo requires internet access.**

            Connection error: %s
        """
            % e.reason
        )

```

--------------------------------

### Add OpenAI API Key Input to Streamlit Sidebar

Source: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/llm-quickstart

Creates a password-type text input field in the Streamlit sidebar for the user to enter their OpenAI API key. This keeps the key input separate from the main app interface.

```python
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
```

--------------------------------

### Columns Layout

Source: https://docs.streamlit.io/develop/quick-reference/cheat-sheet

Create multi-column layouts to arrange elements side-by-side.

```APIDOC
## Columns

### Description
Organize your app's layout into columns for better structure and presentation.

### Method
Python

### Endpoint
N/A

### Parameters
None

### Request Example
```python
# Two equal columns:
col1, col2 = st.columns(2)
col1.write("This is column 1")
col2.write("This is column 2")

# Three different columns:
col1, col2, col3 = st.columns([3, 1, 1])
# col1 is larger.

# Bottom-aligned columns
col1, col2 = st.columns(2, vertical_alignment="bottom")

# You can also use "with" notation:
with col1:
    st.radio("Select one:", [1, 2])
```

### Response
N/A
```

--------------------------------

### Streamlit App to Query Neon Database

Source: https://docs.streamlit.io/develop/tutorials/databases/neon

Python code for a Streamlit application that connects to a Neon database using `st.connection`, queries a 'home' table, and displays the results. It demonstrates basic Streamlit SQL connection usage and data display.

```python
# streamlit_app.py

import streamlit as st

# Initialize connection.
conn = st.connection("neon", type="sql")

# Perform query.
df = conn.query('SELECT * FROM home;', ttl="10m")

# Print results.
for row in df.itertuples():
    st.write(f"{row.name} has a :{row.pet}:")
```

--------------------------------

### Plot Maps with st.map()

Source: https://docs.streamlit.io/get-started/fundamentals/main-concepts

Illustrates how to plot data points on a map using st.map(). The function expects a DataFrame with 'lat' and 'lon' columns to specify the coordinates for the map markers.

```python
import streamlit as st
import numpy as np
import pandas as pd

map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)
```

--------------------------------

### Cache ML Model Loading with st.cache_resource

Source: https://docs.streamlit.io/develop/concepts/architecture/caching

Demonstrates how to use `st.cache_resource` to load and cache an ML model. The model is loaded only once and reused across sessions, improving performance and saving memory. It includes user input for text classification.

```python
from transformers import pipeline

@st.cache_resource  # 👇 Add the caching decorator
def load_model():
    return pipeline("sentiment-analysis")

model = load_model()

query = st.text_input("Your query", value="I love Streamlit! 🎈")
if query:
    result = model(query)[0]  # 👇 Classify the query text
    st.write(result)
```

--------------------------------

### Configure Code Font Size with theme.codeFontSize

Source: https://docs.streamlit.io/develop/quick-reference/release-notes/2025

Enables setting the font size for code displayed in `st.code`, `st.json`, and `st.help` widgets. This is controlled by the `theme.codeFontSize` configuration option in `config.toml`, allowing developers to adjust code readability.

```toml
[theme]
codeFontSize = "14px"
```

--------------------------------

### Build Docker Image for Streamlit App

Source: https://docs.streamlit.io/deploy/tutorials/docker

Builds a Docker image from a Dockerfile in the current directory. The `-t` flag tags the image with a specified name, 'streamlit' in this case. This image can then be used to run the Streamlit application in a container.

```docker
docker build -t streamlit .
```

--------------------------------

### Built-in Connections

Source: https://docs.streamlit.io/develop/api-reference/connections

Information on using Streamlit's pre-defined connections for Snowflake and SQL databases.

```APIDOC
## GET /websites/streamlit_io/connections/built_in

### Description
Utilize Streamlit's built-in connection types for common data sources like Snowflake and SQL databases.

### Method
GET

### Endpoint
`/websites/streamlit_io/connections/built_in`

### Parameters
None

### Request Example
```python
# Snowflake Connection
conn_snowflake = st.connection('snowflake')

# SQL Connection
conn_sql = st.connection('sql')
```

### Response
#### Success Response (200)
Provides access to pre-configured connection objects.

#### Response Example
```json
{
  "message": "Built-in connections are available for use."
}
```
```

--------------------------------

### Streamlit CLI: New App Initialization

Source: https://docs.streamlit.io/develop/api-reference/cli

Command to generate the basic file structure for a new Streamlit application project.

```bash
streamlit init
```

--------------------------------

### Configure Local Streamlit Secrets for MongoDB

Source: https://docs.streamlit.io/develop/tutorials/databases/mongodb

This TOML configuration file defines the connection parameters for a local MongoDB instance. It should be placed in the `.streamlit/secrets.toml` file in your Streamlit app's root directory. Sensitive information like username and password should be replaced with actual credentials.

```toml
# .streamlit/secrets.toml

[mongo]
host = "localhost"
port = 27017
username = "xxx"
password = "xxx"
```

--------------------------------

### Directory Structure for Streamlit App

Source: https://docs.streamlit.io/develop/tutorials/configuration-and-theming/external-fonts

Illustrates the expected file and directory structure for a Streamlit project using custom configurations, including the `.streamlit` directory for the `config.toml` file.

```text
your_repository/
├── .streamlit/
│   └── config.toml
└── streamlit_app.py
```

--------------------------------

### Group Streamlit Pages into Lists

Source: https://docs.streamlit.io/develop/tutorials/multipage/dynamic-navigation

Groups previously defined Streamlit pages into lists for organization. These lists are used to construct the navigation menu.

```python
account_pages = [logout_page, settings]
request_pages = [request_1, request_2]
respond_pages = [respond_1, respond_2]
admin_pages = [admin_1, admin_2]
```

--------------------------------

### Running Streamlit Apps from GitHub and Gist URLs

Source: https://docs.streamlit.io/develop/quick-reference/release-notes/2019

Allows direct execution of Streamlit apps by providing GitHub repository or Gist URLs to the `streamlit run` command. This simplifies the process of running shared Streamlit applications.

```bash
streamlit run https://github.com/user/repo/blob/main/app.py
```

```bash
streamlit run https://gist.github.com/user/gist_id
```

--------------------------------

### AppTest - Basic Usage

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=tutorials&slug=llms+&slug=build-conversational-apps

Demonstrates basic usage of AppTest for simulating a Streamlit app and performing tests.

```APIDOC
## App Testing with AppTest

### Description
`st.testing.v1.AppTest` simulates a running Streamlit app for testing.

### Method
Python

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("streamlit_app.py")
at.secrets["WORD"] = "Foobar"
at.run()
assert not at.exception

at.text_input("word").input("Bazbat").run()
assert at.warning[0].value == "Try again."
```

### Response
#### Success Response (200)
N/A

#### Response Example
N/A
```

--------------------------------

### Display DataFrames with st.write()

Source: https://docs.streamlit.io/get-started/fundamentals/main-concepts

Demonstrates how to use st.write() to display a Pandas DataFrame. st.write() is a versatile function that can render various data types, including DataFrames, automatically determining the best way to display them.

```python
import streamlit as st
import pandas as pd

st.write("Here's our first attempt at using data to create a table:")
st.write(pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
}))
```

--------------------------------

### Personalize Greeting After Login in Streamlit

Source: https://docs.streamlit.io/develop/tutorials/authentication/google

This Python code snippet replaces the generic user display with a personalized welcome message. After a user successfully logs in, it uses `st.user.name` to greet them by name, enhancing the user experience. This demonstrates how to access and utilize user-specific data after authentication.

```python
else:
    st.header(f"Welcome, {st.user.name}!")
```

--------------------------------

### Draw Line Charts with st.line_chart()

Source: https://docs.streamlit.io/get-started/fundamentals/main-concepts

Shows how to create a line chart from a Pandas DataFrame using st.line_chart(). This function is suitable for visualizing time-series data or trends.

```python
import streamlit as st
import numpy as np
import pandas as pd

chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)
```

--------------------------------

### Secrets Management - Singleton

Source: https://docs.streamlit.io/develop/api-reference/connections

Accessing secrets using the `st.secrets` singleton, typically from a TOML file.

```APIDOC
## GET /websites/streamlit_io/secrets/singleton

### Description
Access secrets stored in a local TOML file using the `st.secrets` singleton object.

### Method
GET

### Endpoint
`/websites/streamlit_io/secrets/singleton`

### Parameters
None

### Request Example
```python
# Accessing a secret named 'OpenAI_key'
key = st.secrets["OpenAI_key"]
```

### Response
#### Success Response (200)
Returns the value of the requested secret.

#### Response Example
```json
{
  "secret_value": "<YOUR_SECRET_KEY>"
}
```
```

--------------------------------

### Run Streamlit Docker Container

Source: https://docs.streamlit.io/deploy/tutorials/docker

Runs a Docker container from the 'streamlit' image. The `-p` flag maps port 8501 on the host machine to port 8501 inside the container, making the Streamlit app accessible via a web browser.

```docker
docker run -p 8501:8501 streamlit
```

--------------------------------

### Import Libraries for Streamlit App

Source: https://docs.streamlit.io/get-started/tutorials/create-an-app

Initializes a Streamlit application by importing necessary Python libraries: streamlit for app functionalities, pandas for data manipulation, and numpy for numerical operations. These are fundamental for building interactive data applications.

```python
import streamlit as st
import pandas as pd
import numpy as np
```

--------------------------------

### Use Streamlit Widgets and Control Interactivity

Source: https://docs.streamlit.io/develop/quick-reference/cheat-sheet

Demonstrates how to use Streamlit widgets like number_input, selectbox, and slider, and how to disable widgets to remove interactivity. It shows how to capture widget return values in variables and display them.

```python
for i in range(int(st.number_input("Num:"))):
    foo()
if st.sidebar.selectbox("I:",["f"]) == "f":
    b()
my_slider_val = st.slider("Quinn Mallory", 1, 88)
st.write(slider_val)
st.slider("Pick a number", 0, 100, disabled=True)
```

--------------------------------

### Add Subtitles to Streamlit Video

Source: https://docs.streamlit.io/develop/api-reference/media/st

This snippet demonstrates how to display a video from a URL and associate a VTT subtitle file with it. The subtitles are automatically enabled for the viewer. Ensure the VTT file is accessible.

```python
import streamlit as st

VIDEO_URL = "https://example.com/not-youtube.mp4"
st.video(VIDEO_URL, subtitles="subtitles.vtt")
```

--------------------------------

### Interacting with App Elements

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=configuration&slug=chat

Demonstrates how to interact with various Streamlit elements using `AppTest`.

```APIDOC
## Interacting with App Elements

### Description
`AppTest` provides representations for various Streamlit elements, allowing you to simulate user interactions and assert their states.

### Element Types

#### Block
Represents container elements like `st.chat_message`, `st.columns`, `st.sidebar`, `st.tabs`, and the main app body.
```python
# at.sidebar returns a Block
at.sidebar.button[0].click().run()
assert not at.exception
```

#### Element
The base class for all elements, including `st.title`, `st.header`, `st.markdown`, `st.dataframe`.
```python
# at.title returns a sequence of Title
# Title inherits from Element
assert at.title[0].value == "My awesome app"
```

#### Button
Represents `st.button` and `st.form_submit_button`.
```python
at.button[0].click().run()
```

#### ChatInput
Represents `st.chat_input`.
```python
at.chat_input[0].set_value("What is Streamlit?").run()
```

#### Checkbox
Represents `st.checkbox`.
```python
at.checkbox[0].check().run()
```

#### ColorPicker
Represents `st.color_picker`.
```python
at.color_picker[0].pick("#FF4B4B").run()
```

#### DateInput
Represents `st.date_input`.
```python
import datetime
release_date = datetime.date(2023, 10, 26)
at.date_input[0].set_value(release_date).run()
```

#### Multiselect
Represents `st.multiselect`.
```python
at.multiselect[0].select("New York").run()
```

#### NumberInput
Represents `st.number_input`.
```python
at.number_input[0].increment().run()
```

#### Radio
Represents `st.radio`.
```python
at.radio[0].set_value("New York").run()
```

#### SelectSlider
Represents `st.select_slider`.
```python
at.select_slider[0].set_range("A","C").run()
```

#### Selectbox
Represents `st.selectbox`.
```python
at.selectbox[0].select("New York").run()
```

#### Slider
Represents `st.slider`.
```python
at.slider[0].set_range(2,5).run()
```

#### TextArea
Represents `st.text_area`.
```python
at.text_area[0].input("Streamlit is awesome!").run()
```

#### TextInput
Represents `st.text_input`.
```python
at.text_input[0].input("Streamlit").run()
```

#### TimeInput
Represents `st.time_input`.
```python
at.time_input[0].increment().run()
```

#### Toggle
Represents `st.toggle`.
```python
at.toggle[0].set_value("True").run()
```
```

--------------------------------

### AppTest.from_string

Source: https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1

Create an instance of `AppTest` to simulate an app page defined within a string. Useful for testing short scripts inline.

```APIDOC
## AppTest.from_string

### Description
Create an instance of `AppTest` to simulate an app page defined within a string. This is useful for testing short scripts that fit comfortably as an inline string in the test itself, without having to create a separate file for it. The script must be executable on its own and so must contain all necessary imports.

### Method
POST

### Endpoint
/websites/streamlit_io/AppTest/from_string

### Parameters
#### Path Parameters
- **script** (str) - Required - The string contents of the script to be run.
- **default_timeout** (float) - Optional - Default time in seconds before a script run is timed out. Can be overridden for individual `.run()` calls.

### Request Body
```json
{
  "script": "import streamlit as st\nst.write('Hello, world!')",
  "default_timeout": 5.0
}
```

### Response
#### Success Response (200)
- **AppTest Instance** (AppTest) - A simulated Streamlit app for testing. The simulated app can be executed via `.run()`.

#### Response Example
```json
{
  "message": "AppTest instance created successfully"
}
```
```

--------------------------------

### Define Function to Generate AI Response

Source: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/llm-quickstart

Defines a Python function that initializes the ChatOpenAI model with a specified temperature and the user's API key. It then invokes the model with the provided input text and displays the response using `st.info`.

```python
def generate_response(input_text):
    model = ChatOpenAI(temperature=0.7, api_key=openai_api_key)
    st.info(model.invoke(input_text))
```

--------------------------------

### Use Widgets with Keys and Session State

Source: https://docs.streamlit.io/get-started/fundamentals/main-concepts

Explains how to assign a unique key to a widget, such as st.text_input(), which automatically stores its state in Streamlit's session state. This allows accessing widget values programmatically.

```python
import streamlit as st
st.text_input("Your name", key="name")

# You can access the value at any point with:
st.session_state.name
```

--------------------------------

### Display a camera input widget in Python

Source: https://docs.streamlit.io/develop/api-reference/widgets

Enables users to capture images directly using their device's camera within the Streamlit app. Returns the captured image data. Requires user permission to access the camera.

```python
image = st.camera_input("Take a picture")
```

--------------------------------

### Display Lottie Animation in Streamlit (Python)

Source: https://docs.streamlit.io/develop/api-reference/media

This snippet demonstrates how to load a Lottie animation from a URL and display it in a Streamlit application. It requires the Streamlit library and assumes the `load_lottieurl` and `st_lottie` functions are available.

```python
lottie_hello = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_V9t630.json")

st_lottie(lottie_hello, key="hello")
```

--------------------------------

### Log in a user with st.login() in Python

Source: https://docs.streamlit.io/develop/api-reference/user

The `st.login()` function initiates an authentication flow with an identity provider. This is the primary method for users to log into a Streamlit application. It does not take any arguments.

```python
st.login()
```

--------------------------------

### Streamlit CLI Command

Source: https://docs.streamlit.io/develop/tutorials/execution-flow/create-a-multiple-container-fragment

This command initializes and runs a Streamlit application from the terminal. It navigates to the project directory and executes the Streamlit script, making the app accessible via a web browser.

```bash
streamlit run app.py
```

--------------------------------

### Clone Public Git Repository in Docker

Source: https://docs.streamlit.io/deploy/tutorials/docker

Clones the contents of a public Git repository into the current working directory of the Docker container. This is used to fetch the application's source code.

```dockerfile
RUN git clone https://github.com/streamlit/streamlit-example.git .
```

--------------------------------

### Display Progress Bar in Streamlit (Python)

Source: https://docs.streamlit.io/develop/api-reference/widgets

Provides a simple way to integrate a progress bar into Streamlit applications, often used to indicate the progress of long-running tasks. It requires the `stqdm` library and a loop to update the progress.

```python
from stqdm import stqdm
from time import sleep

for _ in stqdm(range(50)):
    sleep(0.5)
```

--------------------------------

### Create Streamlit Form for User Input and Submission

Source: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/llm-quickstart

Implements a Streamlit form containing a text area for user input and a submit button. It includes validation to check if the API key is provided and calls the `generate_response` function upon successful submission.

```python
with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "What are the three key pieces of advice for learning how to code?",
    )
    submitted = st.form_submit_button("Submit")
    if not openai_api_key.startswith("sk-"):
        st.warning("Please enter your OpenAI API key!", icon="⚠")
    if submitted and openai_api_key.startswith("sk-"):
        generate_response(text)
```

--------------------------------

### Enable Static File Serving in Streamlit Config

Source: https://docs.streamlit.io/develop/concepts/configuration/serving-static-files

This configuration enables the static file serving feature in Streamlit. It should be added under the `[server]` section of your `.streamlit/config.toml` file. This allows Streamlit apps to host and serve small, static media files.

```toml
# .streamlit/config.toml

[server]
enableStaticServing = true
```

--------------------------------

### Connect to Snowflake from Streamlit Community Cloud

Source: https://docs.streamlit.io/develop/tutorials/databases/snowflake

This section outlines the additional steps for connecting Streamlit apps hosted in Community Cloud to Snowflake. It emphasizes the need for a `requirements.txt` file listing dependencies like `snowflake-snowpark-python` and configuring secrets using the `.streamlit/secrets.toml` format.

```text
# requirements.txt
snowflake-snowpark-python
streamlit

# .streamlit/secrets.toml
[connections.snowflake]
account = "your_account"
user = "your_user"
password = "your_password"
role = "your_role"
warehouse = "your_warehouse"
database = "your_database"
schema = "your_schema"
```

--------------------------------

### Display a date input widget in Python

Source: https://docs.streamlit.io/develop/api-reference/widgets

Renders a date picker for users to select a specific date. Returns a `datetime.date` object. Useful for collecting date-related information.

```python
date = st.date_input("Your birthday")
```

--------------------------------

### Build Streamlit LLM App with LangChain and OpenAI

Source: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/llm-quickstart

The core Python code for a Streamlit application that interacts with OpenAI's API via LangChain to generate text. It includes a title, API key input, and a form for user prompts.

```python
import streamlit as st
from langchain_openai.chat_models import ChatOpenAI

st.title("🦜🔗 Quickstart App")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")


def generate_response(input_text):
    model = ChatOpenAI(temperature=0.7, api_key=openai_api_key)
    st.info(model.invoke(input_text))


with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "What are the three key pieces of advice for learning how to code?",
    )
    submitted = st.form_submit_button("Submit")
    if not openai_api_key.startswith("sk-"):
        st.warning("Please enter your OpenAI API key!", icon="⚠")
    if submitted and openai_api_key.startswith("sk-"):
        generate_response(text)
```

--------------------------------

### Streamlit Command Line Interface Commands

Source: https://docs.streamlit.io/develop/quick-reference/cheat-sheet

A collection of common Streamlit commands that can be executed from the command line to manage the Streamlit environment, cache, configuration, and application execution.

```python
streamlit cache clear
streamlit config show
streamlit docs
streamlit hello
streamlit help
streamlit init
streamlit run streamlit_app.py
streamlit version
```

--------------------------------

### Display Daily and Monthly Sales Side-by-Side in Streamlit

Source: https://docs.streamlit.io/develop/tutorials/execution-flow/trigger-a-full-script-rerun-from-a-fragment

Creates two columns in the Streamlit app and calls functions to display daily sales in the left column and monthly sales in the right column. This allows for a comparative view of sales data.

```Python
import streamlit as st

# Assuming show_daily_sales and show_monthly_sales functions are defined elsewhere
# Assuming get_data() function is defined elsewhere

data = get_data()
daily, monthly = st.columns(2)
with daily:
    show_daily_sales(data)
with monthly:
    show_monthly_sales(data)
```

--------------------------------

### Create Mentions with Streamlit Extras (Community Component)

Source: https://docs.streamlit.io/develop/api-reference/text

A utility from the 'Streamlit Extras' library to create clickable mention links with icons and URLs. Useful for linking to external resources or profiles. Requires the `streamlit-extras` library.

```python
mention(label="An awesome Streamlit App", icon="streamlit",  url="https://extras.streamlit.app",)
```

--------------------------------

### Retrieve Feedback for Assistant Messages in Streamlit

Source: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/chat-response-feedback

Checks if a message role is 'assistant' and retrieves any associated user feedback. If no feedback is present, it defaults to None, allowing for conditional display or handling.

```python
if message["role"] == "assistant":
    feedback = message.get("feedback", None)
```

--------------------------------

### Selectbox for Options in Streamlit

Source: https://docs.streamlit.io/get-started/fundamentals/main-concepts

Illustrates the use of `st.selectbox` to allow users to choose an option from a predefined list. The selected option is then displayed in the app. This is useful for user input and filtering.

```python
import streamlit as st
import pandas as pd

df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
    })

option = st.selectbox(
    'Which number do you like best?',
     df['first column'])

'You selected: ', option
```

--------------------------------

### AppTest.switch_page

Source: https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1

Switch to another page of the app. A follow-up `AppTest.run()` call is needed to see the elements on the new page.

```APIDOC
## AppTest.switch_page

### Description
Switch to another page of the app. This method does not automatically rerun the app. Use a follow-up call to `AppTest.run()` to obtain the elements on the selected page.

### Method
POST

### Endpoint
/websites/streamlit_io/AppTest/switch_page

### Parameters
#### Path Parameters
- **page_path** (str) - Required - Path of the page to switch to. The path must be relative to the main script's location (e.g. "pages/my_page.py").

### Request Body
```json
{
  "page_path": "pages/another_page.py"
}
```

### Response
#### Success Response (200)
- **AppTest Instance** (AppTest) - self

#### Response Example
```json
{
  "message": "Page switched successfully"
}
```
```

--------------------------------

### Streamlit App Logic and Configuration

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

Documentation for managing app logic, including user authentication, navigation between pages, and controlling script execution flow.

```APIDOC
## App Logic and Configuration

### Description
This section covers essential Streamlit features for structuring your application, managing user sessions, and controlling the execution flow.

### Authentication and User Info

#### Log in a user
Initiates an authentication flow with an identity provider.
```python
st.login()
```

#### Log out a user
Removes a user's identity information from the session.
```python
st.logout()
```

#### User Info
Provides access to information about the currently logged-in user.
```python
if st.user.is_logged_in:
  st.write(f"Welcome back, {st.user.name}!")
```

### Navigation and Pages

#### Navigation
Configures the available pages in a multipage Streamlit application.
```python
st.navigation({
    "Your account": [log_out, settings],
    "Reports": [overview, usage],
    "Tools": [search]
})
```

#### Page Definition
Defines a page within a multipage application, specifying its source file, title, and icon.
```python
home = st.Page(
    "home.py",
    title="Home",
    icon=":material/home:"
)
```

#### Page Link
Creates a navigable link to another page within the application.
```python
st.page_link("app.py", label="Home", icon="🏠")
st.page_link("pages/profile.py", label="My profile")
```

#### Switch Page
Programmatically navigates the user to a specified page.
```python
st.switch_page("pages/my_page.py")
```

### Execution Flow

#### Modal Dialog
Inserts a modal dialog that can execute independently.
```python
@st.dialog("Sign up")
def email_form():
    name = st.text_input("Name")
    email = st.text_input("Email")
```

#### Forms
Groups elements together with a submit button to batch input processing.
```python
with st.form(key='my_form'):
    name = st.text_input("Name")
    email = st.text_input("Email")
    st.form_submit_button("Sign up")
```

#### Fragments
Defines a section of the app that can rerun independently.
```python
@st.fragment(run_every="10s")
def fragment():
    df = get_data()
    st.line_chart(df)
```

#### Rerun Script
Immediately reruns the current Streamlit script.
```python
st.rerun()
```

#### Stop Execution
Halts the script execution immediately.
```python
st.stop()
```
```

--------------------------------

### Show multiple stacked toasts with Streamlit st.toast

Source: https://docs.streamlit.io/develop/api-reference/status/st

Illustrates how to display multiple toast messages sequentially, which will stack in the UI. It also shows how hovering over a toast can pause its disappearance. This is useful for multi-step processes.

```python
import time
import streamlit as st

if st.button("Three cheers"):
    st.toast("Hip!")
    time.sleep(0.5)
    st.toast("Hip!")
    time.sleep(0.5)
    st.toast("Hooray!", icon="🎉")
```

--------------------------------

### Display Naive Datetime in Streamlit

Source: https://docs.streamlit.io/develop/concepts/design/timezone-handling

Demonstrates how Streamlit displays a naive datetime instance (without timezone information) on the frontend. The output will be the same as the backend representation, without any timezone adjustments.

```python
import streamlit as st
from datetime import datetime

st.write(datetime(2020, 1, 10, 10, 30))
# Outputs: 2020-01-10 10:30:00
```

--------------------------------

### Streamlit CLI: Help Information

Source: https://docs.streamlit.io/develop/api-reference/cli

Command to display a list of all available Streamlit CLI commands and their brief descriptions.

```bash
streamlit help
```

--------------------------------

### Initialize AppTest from String

Source: https://docs.streamlit.io/develop/api-reference/app-testing

Initializes a simulated Streamlit app for testing from a Python script provided as a string. It supports setting secrets and running the app to verify its behavior.

```Python
from streamlit.testing.v1 import AppTest

app_script = """
import streamlit as st

word_of_the_day = st.text_input("What's the word of the day?", key="word")
if word_of_the_day == st.secrets["WORD"]:
    st.success("That's right!")
elif word_of_the_day and word_of_the_day != st.secrets["WORD"]:
    st.warn("Try again.")
"""

at = AppTest.from_string(app_script)
at.secrets["WORD"] = "Foobar"
at.run()
assert not at.exception
```

--------------------------------

### Implement Multi-Page Apps in Streamlit

Source: https://docs.streamlit.io/develop/api-reference/execution-flow

Uses the st-pages library by @blackary for an experimental implementation of Streamlit Multi-Page Apps. It allows defining pages and their configurations.

```python
from st_pages import Page, show_pages, add_page_title

show_pages([
    Page("streamlit_app.py", "Home", "🏠"),
    Page("other_pages/page2.py", "Page 2", ":books:"),
])

add_page_title()
```

--------------------------------

### Display a link button widget in Python

Source: https://docs.streamlit.io/develop/api-reference/widgets

Creates a button that functions as a hyperlink, navigating the user to a specified URL. Requires a URL string as an argument. The button's label is also configurable.

```python
st.link_button("Go to gallery", url)
```

--------------------------------

### Display Data Loading Status in Streamlit

Source: https://docs.streamlit.io/get-started/tutorials/create-an-app

Shows a 'Loading data...' message in the Streamlit app while data is being fetched and processed. Once the data is loaded, the message is updated to 'Loading data...done!', providing user feedback during potentially long operations.

```python
# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_data(10000)
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')
```

--------------------------------

### Log In a User with Streamlit

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

Initiates an authentication flow with an identity provider using `st.login()`. This function is the entry point for user authentication in Streamlit apps.

```python
st.login()
```

--------------------------------

### Progress Bar for Long Computations in Streamlit

Source: https://docs.streamlit.io/get-started/fundamentals/main-concepts

Explains how to use `st.progress` to display the status of long-running computations in real-time. It includes simulating a delay with `time.sleep` and updating a progress bar and text iteratively.

```python
import streamlit as st
import time

'Starting a long computation...'

# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Iteration {i+1}')
  bar.progress(i + 1)
  time.sleep(0.1)

'...and now we're done!'
```

--------------------------------

### Add User Prompt to Chat History in Streamlit

Source: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/chat-response-feedback

Accepts user input from `st.chat_input`, displays it in a chat message, and appends it to the session state's chat history. This code snippet demonstrates basic chat input handling in Streamlit.

```Python
if prompt := st.chat_input("Say something"):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.history.append({"role": "user", "content": prompt})
```

```Python
prompt = st.chat_input("Say something")
if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.history.append({"role": "user", "content": prompt})
```

--------------------------------

### Test Multipage Streamlit App with Session State

Source: https://docs.streamlit.io/develop/concepts/app-testing/beyond-the-basics

Shows how to test a specific page of a multipage Streamlit application by manually setting the `session_state` before running the app. This simulates user interaction and data persistence across pages.

```Python
"""test_second.py"""
from streamlit.testing.v1 import AppTest

def test_balloons():
    at = AppTest.from_file("pages/second.py")
    at.session_state["magic_word"] = "Balloons"
    at.run()
    assert at.markdown[0].value == ":balloon:"
```

--------------------------------

### Define Available User Roles in Streamlit

Source: https://docs.streamlit.io/develop/tutorials/multipage/dynamic-navigation

This Python code defines a list named `ROLES` containing possible user roles for a Streamlit application, including `None` for unauthenticated users, 'Requester', 'Responder', and 'Admin'. This list is used for role selection, likely in a login or authentication process.

```python
ROLES = [None, "Requester", "Responder", "Admin"]
```

--------------------------------

### Secrets Management - File Structure

Source: https://docs.streamlit.io/develop/api-reference/connections

Details on how to structure your secrets file (e.g., `secrets.toml`) for local storage.

```APIDOC
## POST /websites/streamlit_io/secrets/file

### Description
Store your secrets in a per-project or per-profile TOML file (e.g., `secrets.toml`).

### Method
POST

### Endpoint
`/websites/streamlit_io/secrets/file`

### Parameters
#### Request Body
- **secret_name** (string) - Required - The name of the secret to be stored.
- **secret_value** (string) - Required - The value of the secret.

### Request Example
```toml
# secrets.toml
OpenAI_key = "<YOUR_SECRET_KEY>"
```

### Response
#### Success Response (200)
Indicates that the secrets file structure is correctly configured.

#### Response Example
```json
{
  "message": "Secrets file configured successfully."
}
```
```

--------------------------------

### Streamlit Menu Functions (Python)

Source: https://docs.streamlit.io/develop/tutorials/multipage/st

Defines helper functions to render navigation menus for different user roles in a Streamlit app. Includes logic for authenticated users, unauthenticated users, and redirection based on login status. Uses `st.session_state` to manage user roles.

```Python
import streamlit as st


def authenticated_menu():
    # Show a navigation menu for authenticated users
    st.sidebar.page_link("app.py", label="Switch accounts")
    st.sidebar.page_link("pages/user.py", label="Your profile")
    if st.session_state.role in ["admin", "super-admin"]:
        st.sidebar.page_link("pages/admin.py", label="Manage users")
        st.sidebar.page_link(
            "pages/super-admin.py",
            label="Manage admin access",
            disabled=st.session_state.role != "super-admin",
        )

def unauthenticated_menu():
    # Show a navigation menu for unauthenticated users
    st.sidebar.page_link("app.py", label="Log in")

def menu():
    # Determine if a user is logged in or not, then show the correct
    # navigation menu
    if "role" not in st.session_state or st.session_state.role is None:
        unauthenticated_menu()
        return
    authenticated_menu()

def menu_with_redirect():
    # Redirect users to the main page if not logged in, otherwise continue to
    # render the navigation menu
    if "role" not in st.session_state or st.session_state.role is None:
        st.switch_page("app.py")
    menu()

```

--------------------------------

### Add Title and Description to Streamlit App

Source: https://docs.streamlit.io/develop/tutorials/execution-flow/trigger-a-full-script-rerun-from-a-fragment

Adds a main title and a descriptive markdown text to the Streamlit application. This helps in providing context and information about the app's purpose to the user.

```Python
import streamlit as st

st.title("Daily vs monthly sales, by product")
st.markdown("This app shows the 2023 daily sales for Widget A through Widget Z.")
```

--------------------------------

### Streamlit: Stateful Sidebar Widgets in Entrypoint File

Source: https://docs.streamlit.io/develop/concepts/multipage-apps/widgets

Demonstrates how to define stateful widgets (selectbox and slider) in the sidebar of a Streamlit multipage app's entrypoint file. This ensures widgets remain stateful across all pages by associating them with the entrypoint instead of individual pages. Requires `st.Page` and `st.navigation`.

```Python
import streamlit as st

pg = st.navigation([st.Page("page_1.py"), st.Page("page_2.py")])

st.sidebar.selectbox("Group", ["A","B","C"], key="group")
st.sidebar.slider("Size", 1, 5, key="size")

pg.run()
```

--------------------------------

### Define Streamlit Session State Role

Source: https://docs.streamlit.io/develop/tutorials/multipage/dynamic-navigation

Saves the user's role from Streamlit's session state to a local variable for easier access. This is a common first step in role-based navigation.

```python
role = st.session_state.role
```

--------------------------------

### Cache PyTorch Model Loading with st.cache_resource

Source: https://docs.streamlit.io/develop/concepts/architecture/caching

Illustrates caching a PyTorch model using `st.cache_resource`. This is beneficial for large models that are time-consuming to load, ensuring they are loaded only once and reused efficiently across the application.

```python
@st.cache_resource
def load_model():
    model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.eval()
    return model

model = load_model()
```

--------------------------------

### Create Tabs with Context Management in Streamlit

Source: https://docs.streamlit.io/develop/api-reference/layout/st

Demonstrates how to create tabs using Streamlit's st.tabs function and populate them with content using the 'with' context manager. This is the preferred method for adding elements to tabs.

```python
import streamlit as st

tab1, tab2, tab3 = st.tabs(["Cat", "Dog", "Owl"])

with tab1:
    st.header("A cat")
    st.image("https://static.streamlit.io/examples/cat.jpg", width=200)
with tab2:
    st.header("A dog")
    st.image("https://static.streamlit.io/examples/dog.jpg", width=200)
with tab3:
    st.header("An owl")
    st.image("https://static.streamlit.io/examples/owl.jpg", width=200)
```

--------------------------------

### Create a Compatible Generator Wrapper for Unsupported Streams

Source: https://docs.streamlit.io/develop/api-reference/write-magic/st

Illustrates how to create a wrapper function to make an unsupported stream object compatible with st.write_stream. This involves iterating through the unsupported stream and yielding processed chunks.

```python
for chunk in unsupported_stream:
    yield preprocess(chunk)
```

--------------------------------

### Streamlit layout for no errors scenario

Source: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/validate-and-edit-chat-responses

Sets up Streamlit columns for displaying action buttons when no errors are present in the validation list. This prepares the UI for subsequent actions like accepting the result.

```python
else:
    cols = st.columns(2)

```

--------------------------------

### Apply Text Mining with NLU (Community Component)

Source: https://docs.streamlit.io/develop/api-reference/text

A third-party component for applying Natural Language Understanding (NLU) tasks, such as sentiment analysis, to text. It requires the `nlu` library and potentially pre-trained models. The input is typically a string.

```python
nlu.load('sentiment').predict('I love NLU! <3')
```

--------------------------------

### Configure Snowflake Connection Parameters in secrets.toml

Source: https://docs.streamlit.io/develop/tutorials/databases/snowflake

This TOML snippet shows how to define Snowflake connection parameters, including account, user, private key file, role, warehouse, database, and schema, within the `.streamlit/secrets.toml` file for local Streamlit app secrets management. It uses key-pair authentication.

```toml
[connections.snowflake]
account = "xxxxxxx-xxxxxxx"
user = "xxx"
private_key_file = "../xxx/xxx.p8"
role = "xxx"
warehouse = "xxx"
database = "xxx"
schema = "xxx"
```

--------------------------------

### Configure Client Error Details (TOML)

Source: https://docs.streamlit.io/develop/concepts/configuration/options

This TOML snippet illustrates how to configure whether detailed error messages are shown to the client. This setting is part of the `[client]` section in configuration files.

```toml
[client]
showErrorDetails = true

```

--------------------------------

### Initialize AppTest from String (Python)

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

Initializes `st.testing.v1.AppTest` from a string containing Streamlit app code. This is useful for testing dynamically generated or short app scripts.

```python
from streamlit.testing.v1 import AppTest

at = AppTest.from_string(app_script_as_string)
at.run()
```

--------------------------------

### Configure Streamlit Page Settings

Source: https://docs.streamlit.io/develop/api-reference/configuration

Explains how to set the default settings for the Streamlit app's page, including the page title and favicon. This function allows for basic customization of the app's appearance in the browser.

```python
st.set_page_config(
  page_title="My app",
  page_icon=":shark:",
)
```

--------------------------------

### Configure Button Radius with theme.buttonRadius

Source: https://docs.streamlit.io/develop/quick-reference/release-notes/2025

Allows customization of the border radius for buttons, separate from other elements. This configuration is set using `theme.buttonRadius` in the `config.toml` file, providing fine-grained control over button aesthetics.

```toml
[theme]
buttonRadius = "8px"
```

--------------------------------

### Streamlit Connection Base Class Implementation

Source: https://docs.streamlit.io/develop/api-reference/connections/st.connections

This Python code demonstrates the structure of a custom Streamlit connection by inheriting from ExperimentalBaseConnection. It requires implementing the _connect method and optionally overriding the scope() method for session-scoped connections.

```python
from streamlit.connections import ExperimentalBaseConnection

class MyConnection(ExperimentalBaseConnection):
    def _connect(self, **kwargs):
        # Implementation to establish the connection
        pass

    @classmethod
    def scope(cls):
        # Return 'session' for session-scoped connections
        return "global"

```

--------------------------------

### Connect to SQLite, Insert, and Query Data in Streamlit

Source: https://docs.streamlit.io/develop/concepts/connections/connecting-to-data

This Python script demonstrates creating a Streamlit SQL connection, defining a table, inserting data, and displaying the queried results in a DataFrame. It utilizes `st.connection()` and `conn.session` for database operations.

```python
# streamlit_app.py

import streamlit as st

# Create the SQL connection to pets_db as specified in your secrets file.
conn = st.connection('pets_db', type='sql')

# Insert some data with conn.session.
with conn.session as s:
    s.execute('CREATE TABLE IF NOT EXISTS pet_owners (person TEXT, pet TEXT);')
    s.execute('DELETE FROM pet_owners;')
    pet_owners = {'jerry': 'fish', 'barbara': 'cat', 'alex': 'puppy'}
    for k in pet_owners:
        s.execute(
            'INSERT INTO pet_owners (person, pet) VALUES (:owner, :pet);',
            params=dict(owner=k, pet=pet_owners[k])
        )
    s.commit()

# Query and display the data you inserted
pet_owners = conn.query('select * from pet_owners')
st.dataframe(pet_owners)
```

--------------------------------

### AppTest Class Initialization

Source: https://docs.streamlit.io/develop/api-reference/app-testing

Initializes a simulated Streamlit app for testing purposes. This can be done by loading the app from a file, a string, or a function.

```APIDOC
## AppTest Class Initialization

### Description
Initializes a simulated Streamlit app for testing purposes. This can be done by loading the app from a file, a string, or a function.

### Methods

#### `AppTest.from_file(file_path: str)`
Initializes a simulated app from a Python file.

**Parameters**

*   **file_path** (str) - Required - The path to the Streamlit app file.

**Request Example**
```python
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("streamlit_app.py")
```

#### `AppTest.from_string(app_script: str)`
Initializes a simulated app from a Python script as a string.

**Parameters**

*   **app_script** (str) - Required - The Python script content of the Streamlit app.

**Request Example**
```python
from streamlit.testing.v1 import AppTest

app_script = """
import streamlit as st

word_of_the_day = st.text_input("What's the word of the day?", key="word")
if word_of_the_day == st.secrets["WORD"]:
    st.success("That's right!")
elif word_of_the_day and word_of_the_day != st.secrets["WORD"]:
    st.warn("Try again.")
"""

at = AppTest.from_string(app_script)
```

#### `AppTest.from_function(app_function: callable)`
Initializes a simulated app from a Python function.

**Parameters**

*   **app_function** (callable) - Required - The function containing the Streamlit app code.

**Request Example**
```python
from streamlit.testing.v1 import AppTest

def app_script (): 
    import streamlit as st

    word_of_the_day = st.text_input("What's the word of the day?", key="word")
    if word_of_the_day == st.secrets["WORD"]:
        st.success("That's right!")
    elif word_of_the_day and word_of_the_day != st.secrets["WORD"]:
        st.warn("Try again.")

at = AppTest.from_function(app_script)
```

### Common Operations

#### Setting Secrets
Secrets can be set directly on the `AppTest` instance.

**Request Example**
```python
at.secrets["WORD"] = "Foobar"
```

#### Running the App
Execute the Streamlit app simulation.

**Request Example**
```python
at.run()
```

#### Asserting No Exceptions
Check if any exceptions occurred during the app run.

**Request Example**
```python
assert not at.exception
```
```

--------------------------------

### Displaying Code Blocks with st.code

Source: https://docs.streamlit.io/develop/api-reference_slug=private-gsheet

Render a code block, optionally with syntax highlighting, using `st.code`.

```APIDOC
## POST /st.code

### Description
Display a code block with optional syntax highlighting.

### Method
POST

### Endpoint
/st.code

### Parameters
#### Request Body
- **code_string** (string) - Required - The code to display.
- **language** (string, optional) - The programming language for syntax highlighting (e.g., 'python').

### Request Example
```json
{
  "code_string": "a = 1234",
  "language": "python"
}
```

### Response
#### Success Response (200)
- **status** (string) - Indicates successful rendering of the code block.

#### Response Example
```json
{
  "status": "success"
}
```
```

--------------------------------

### Configure Local PostgreSQL Secrets for Streamlit

Source: https://docs.streamlit.io/develop/tutorials/databases/postgresql

TOML format for the '.streamlit/secrets.toml' file, specifying PostgreSQL connection details including host, port, database name, username, and password. This file is used for local development and its content should be manually added to Streamlit Community Cloud secrets for deployed apps.

```toml
# .streamlit/secrets.toml

[connections.postgresql]
dialect = "postgresql"
host = "localhost"
port = "5432"
database = "xxx"
username = "xxx"
password = "xxx"
```

--------------------------------

### Display a page link widget in Python

Source: https://docs.streamlit.io/develop/api-reference/widgets

Generates a link to navigate between pages in a multipage Streamlit application. Requires the path to the target page and an optional label and icon. Useful for creating navigation menus.

```python
st.page_link("app.py", label="Home", icon="🏠")
st.page_link("pages/profile.py", label="My profile")
```

--------------------------------

### AppTest - ChatInput Interaction

Source: https://docs.streamlit.io/develop/api-reference_slug=%28&slug=develop&slug=concepts&slug=configuration

Simulates user input into the `st.chat_input` widget.

```APIDOC
## ChatInput

### Description
A representation of `st.chat_input`.

### Method
Python

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
at.chat_input[0].set_value("What is Streamlit?").run()
```

### Response
#### Success Response (200)
N/A

#### Response Example
N/A
```

--------------------------------

### AppTest - ChatInput Interaction

Source: https://docs.streamlit.io/develop/api-reference_slug=%5B...slug%5D

Simulates user input into the `st.chat_input` widget.

```APIDOC
## ChatInput

### Description
A representation of `st.chat_input`.

### Method
Python (AppTest interaction)

### Endpoint
N/A (AppTest method)

### Parameters
None

### Request Example
```python
at.chat_input[0].set_value("What is Streamlit?").run()
```

### Response
#### Success Response (200)
N/A (This is an interaction, not a direct response)

#### Response Example
N/A
```

--------------------------------

### Generate Daily Sales Data for a Year

Source: https://docs.streamlit.io/develop/tutorials/execution-flow/trigger-a-full-script-rerun-from-a-fragment

This Python code generates daily sales figures for each product over a year. It creates a pandas DataFrame, sets a date index from January 1, 2023, to January 1, 2024, and populates it with normally distributed random sales numbers, rounded to two decimal places. The index is then converted to only show dates.

```Python
data = pd.DataFrame({})
sales_dates = np.arange(date(2023, 1, 1), date(2024, 1, 1), timedelta(days=1))
for product, sales in products.items():
    data[product] = np.random.normal(sales, 300, len(sales_dates)).round(2)
data.index = sales_dates
data.index = data.index.date
```

--------------------------------

### Build Function for Random Sales Data Generation

Source: https://docs.streamlit.io/develop/tutorials/execution-flow/trigger-a-full-script-rerun-from-a-fragment

This Python function, decorated with `@st.cache_data`, generates random daily sales data for a year for products 'Widget A' through 'Widget Z'. It utilizes pandas for DataFrame creation, numpy for random number generation, and datetime for date ranges. The caching decorator ensures data is generated only once.

```Python
@st.cache_data
def get_data():
    """Generate random sales data for Widget A through Widget Z"""

    product_names = ["Widget " + letter for letter in string.ascii_uppercase]
    average_daily_sales = np.random.normal(1_000, 300, len(product_names))
    products = dict(zip(product_names, average_daily_sales))

    data = pd.DataFrame({})
    sales_dates = np.arange(date(2023, 1, 1), date(2024, 1, 1), timedelta(days=1))
    for product, sales in products.items():
        data[product] = np.random.normal(sales, 300, len(sales_dates)).round(2)
    data.index = sales_dates
    data.index = data.index.date
    return data
```

--------------------------------

### Manage Streamlit Placeholders, Containers, and Options

Source: https://docs.streamlit.io/develop/quick-reference/cheat-sheet

Covers advanced layout and control features in Streamlit, including replacing elements with `st.empty()`, organizing content with `st.container()`, and configuring app options with `st.set_option()` and `st.set_page_config()`.

```python
# Replace any single element.
element = st.empty()
element.line_chart(...)
element.text_input(...)  # Replaces previous.

# Insert out of order.
elements = st.container()
elements.line_chart(...)
st.write("Hello")
elements.text_input(...)  # Appears above "Hello".

# Horizontal flex
flex = st.container(horizontal=True)
flex.button("A")
flex.button("B")

# Spacing
st.space("small")

st.help(pandas.DataFrame)
st.get_option(key)
st.set_option(key, value)
st.set_page_config(layout="wide")
st.query_params[key]
st.query_params.from_dict(params_dict)
st.query_params.get_all(key)
st.query_params.clear()
st.html("<p>Hi!</p>")
```

--------------------------------

### Streamlit AgGrid Integration

Source: https://docs.streamlit.io/develop/api-reference/data

Integrates the powerful Ag-Grid component into Streamlit applications, offering advanced features like editing, sorting, and filtering. Requires `streamlit-aggrid`.

```python
import pandas as pd
from st_aggrid import AgGrid

df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
grid_return = AgGrid(df, editable=True)

new_df = grid_return['data']
```

--------------------------------

### requirements.txt with Specific Package Versions

Source: https://docs.streamlit.io/deploy/concepts/dependencies

This `requirements.txt` file specifies exact or range-based versions for Streamlit, pandas, and numpy. This is useful for ensuring compatibility and reproducibility across different deployment environments.

```text
streamlit==1.24.1
pandas>2.0
numpy<=1.25.1
```

--------------------------------

### Client Configuration Options in config.toml

Source: https://docs.streamlit.io/develop/api-reference/configuration/config

Shows various client-side configuration options available in config.toml. These settings control how errors are displayed, the visibility of the toolbar, sidebar navigation in multi-page apps, and help links in error messages.

```toml
[client]

# Controls whether uncaught app exceptions and deprecation warnings
# are displayed in the browser. This can be one of the following:
#
# - "full"       : In the browser, Streamlit displays app deprecation
#                  warnings and exceptions, including exception types,
#                  exception messages, and associated tracebacks.
# - "stacktrace" : In the browser, Streamlit displays exceptions,
#                  including exception types, generic exception messages,
#                  and associated tracebacks. Deprecation warnings and
#                  full exception messages will only print to the
#                  console
# - "type"       : In the browser, Streamlit displays exception types and
#                  generic exception messages. Deprecation warnings, full
#                  exception messages, and associated tracebacks only
#                  print to the console.
# - "none"       : In the browser, Streamlit displays generic exception
#                  messages. Deprecation warnings, full exception
#                  messages, associated tracebacks, and exception types
#                  will only print to the console.
# - True         : This is deprecated. Streamlit displays "full"
#                  error details.
# - False        : This is deprecated. Streamlit displays "stacktrace"
#                  error details.
#
# Default: "full"
showErrorDetails = "full"

# Change the visibility of items in the toolbar, options menu,
# and settings dialog (top right of the app).
#
# Allowed values:
# - "auto"      : Show the developer options if the app is accessed through
#                 localhost or through Streamlit Community Cloud as a developer.
#                 Hide them otherwise.
# - "developer" : Show the developer options.
# - "viewer"    : Hide the developer options.
# - "minimal"   : Show only options set externally (e.g. through
#                 Streamlit Community Cloud) or through st.set_page_config.
#                 If there are no options left, hide the menu.
#
# Default: "auto"
toolbarMode = "auto"

# Controls whether to display the default sidebar page navigation in a
# multi-page app. This only applies when app's pages are defined by the
# `pages/` directory.
#
# Default: true
showSidebarNavigation = true

# Controls whether to show external help links (Google, ChatGPT) in
# error displays. The following values are valid:

```

--------------------------------

### Initialize Streamlit Chat History in Session State

Source: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/chat-response-feedback

Initializes an empty list in Streamlit's session state to store chat history if it doesn't already exist. This ensures the chat history is preserved across reruns.

```python
if "history" not in st.session_state:
    st.session_state.history = []
```

--------------------------------

### Display Images in Streamlit

Source: https://docs.streamlit.io/develop/api-reference/media

Demonstrates how to display images in a Streamlit app using various input formats such as NumPy arrays, image bytes, file objects, and URLs. This is useful for visualizing data or embedding static visuals.

```python
st.image(numpy_array)
st.image(image_bytes)
st.image(file)
st.image("https://example.com/myimage.jpg")
```

--------------------------------

### Initialize AppTest from Function

Source: https://docs.streamlit.io/develop/api-reference/app-testing

Initializes a simulated Streamlit app for testing from a Python function that contains the app's logic. This allows for testing apps defined as functions, including setting secrets and running the app.

```Python
from streamlit.testing.v1 import AppTest

def app_script ():
    import streamlit as st

    word_of_the_day = st.text_input("What's the word of the day?", key="word")
    if word_of_the_day == st.secrets["WORD"]:
        st.success("That's right!")
    elif word_of_the_day and word_of_the_day != st.secrets["WORD"]:
        st.warn("Try again.")

at = AppTest.from_function(app_script)
at.secrets["WORD"] = "Foobar"
at.run()
assert not at.exception
```

--------------------------------

### Add PyMySQL Dependencies for TiDB (Terminal)

Source: https://docs.streamlit.io/develop/tutorials/databases/tidb

Lists the Python packages required for Streamlit to connect to TiDB using PyMySQL and SQLAlchemy. This `requirements.txt` entry replaces the mysqlclient dependency with PyMySQL.

```text
# requirements.txt
PyMySQL==x.x.x
SQLAlchemy==x.x.x
```

--------------------------------

### Display a color picker widget in Python

Source: https://docs.streamlit.io/develop/api-reference/widgets

Provides a user interface for selecting a color. Returns the selected color as a string (e.g., '#FF0000'). Allows users to visually choose colors for customization.

```python
color = st.color_picker("Pick a color")
```

--------------------------------

### AppTest - TextInput Interaction

Source: https://docs.streamlit.io/develop/api-reference_slug=%28&slug=develop&slug=concepts&slug=configuration

Simulates inputting text into the `st.text_input` widget.

```APIDOC
## TextInput

### Description
A representation of `st.text_input`.

### Method
Python

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
at.text_input[0].input("Streamlit").run()
```

### Response
#### Success Response (200)
N/A

#### Response Example
N/A
```

--------------------------------

### Initialize Streamlit Session State for Chat History

Source: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/validate-and-edit-chat-responses

Initializes the Streamlit session state variables required for tracking chat stages, conversation history, pending responses, and validation data. This ensures a consistent state across user interactions.

```python
if "stage" not in st.session_state:
    st.session_state.stage = "user"
    st.session_state.history = []
    st.session_state.pending = None
    st.session_state.validation = {}
```

--------------------------------

### Initialize SQLAlchemy SQL Connection in Python

Source: https://docs.streamlit.io/develop/api-reference/connections

Initializes a Streamlit connection object for a SQL database using SQLAlchemy. This provides a standardized way to connect to various SQL databases.

```python
conn = st.connection('sql')
```

--------------------------------

### Streamlit App Configuration for Theme

Source: https://docs.streamlit.io/develop/concepts/architecture/app-chrome

Demonstrates how to set the Streamlit app's page configuration, including theme settings (light, dark, system default) and enabling wide mode. This is typically done at the beginning of a Streamlit script.

```python
import streamlit as st

st.set_page_config(
    page_title="My Streamlit App",
    page_icon=":smiley:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.streamlit.io/library/api-reference/utilities/st.set_page_config',
        'Report a bug': "https://github.com/streamlit/streamlit/issues",
        'About': "This is a Streamlit app example."
    }
)

# Example of theme settings (though typically controlled via UI or config file)
st.markdown("# Welcome to My App!")

# To force wide mode if not set in set_page_config:
# st.set_page_config(layout="wide")

```

--------------------------------

### Configure Database Connection Secrets in TOML

Source: https://docs.streamlit.io/get-started/fundamentals/advanced-concepts

This TOML snippet illustrates how to define connection parameters for a Streamlit app in the `.streamlit/secrets.toml` file. It includes details for a MySQL database connection.

```toml
[connections.my_database]
type="sql"
dialect="mysql"
username="xxx"
password="xxx"
host="example.com"
port=3306
database="mydb"
```

--------------------------------

### Attach Callback to Streamlit Feedback Widget

Source: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/chat-response-feedback

Attaches the `save_feedback` callback function and its required arguments (the message index) to the `st.feedback` widget. This ensures that user interactions with the widget update the chat history.

```python
st.feedback(
    "thumbs",
    key=f"feedback_{i}",
    disabled=feedback is not None,
    on_change=save_feedback,
    args=[i],
)
```

--------------------------------

### Add pyodbc to Streamlit Requirements

Source: https://docs.streamlit.io/develop/tutorials/databases/mssql

Entry for the `requirements.txt` file to include the `pyodbc` Python package, necessary for connecting to SQL Server from Streamlit.

```text
# requirements.txt
pyodbc==x.x.x
```

--------------------------------

### Logger Configuration Options in config.toml

Source: https://docs.streamlit.io/develop/api-reference/configuration/config

Demonstrates how to configure Streamlit's internal logger using the config.toml file. Options include setting the logging level and defining the format for log messages.

```toml
[logger]

# Level of logging for Streamlit's internal logger: "error", "warning",
# "info", or "debug".
#
# Default: "info"
level = "info"

# String format for logging messages. If logger.datetimeFormat is set,
# logger messages will default to `%(asctime)s.%(msecs)03d %(message)s`.
#
# See Python's documentation for available attributes:
# https://docs.python.org/3/library/logging.html#formatter-objects
#
# Default: "% (asctime)s %(message)s"
messageFormat = "% (asctime)s %(message)s"
```

--------------------------------

### Create SQL Database Connection in Python

Source: https://docs.streamlit.io/develop/api-reference/connections

Establishes a connection to a SQL database using Streamlit's connection API and retrieves data. It requires a connection named 'pets_db' of type 'sql'. The output is a Pandas DataFrame.

```python
conn = st.connection('pets_db', type='sql')
pet_owners = conn.query('select * from pet_owners')
st.dataframe(pet_owners)
```

--------------------------------

### Streamlit App to Connect and Display MongoDB Data

Source: https://docs.streamlit.io/develop/tutorials/databases/mongodb

This Python script demonstrates how to connect to a MongoDB database using PyMongo and Streamlit's secrets management. It utilizes Streamlit's caching mechanisms (`st.cache_resource` and `st.cache_data`) to optimize data retrieval and display. Ensure your MongoDB connection details are correctly configured in `.streamlit/secrets.toml`.

```python
# streamlit_app.py

import streamlit as st
import pymongo

# Initialize connection.
# Uses st.cache_resource to only run once.
@st.cache_resource
def init_connection():
    return pymongo.MongoClient(**st.secrets["mongo"])

client = init_connection()

# Pull data from the collection.
# Uses st.cache_data to only rerun when the query changes or after 10 min.
@st.cache_data(ttl=600)
def get_data():
    db = client.mydb
    items = db.mycollection.find()
    items = list(items)  # make hashable for st.cache_data
    return items

items = get_data()

# Print results.
for item in items:
    st.write(f"{item['name']} has a :{item['pet']}:")
```

--------------------------------

### Accessing Script Arguments in Python

Source: https://docs.streamlit.io/develop/api-reference/cli/run

Demonstrates how arguments passed via the `streamlit run` command are accessed within a Python script using `sys.argv`. Note that `sys.argv[0]` is the entrypoint file path.

```python
import sys

print(f'sys.argv[0]: {sys.argv[0]}')
print(f'sys.argv[1]: {sys.argv[1]}')
print(f'sys.argv[2]: {sys.argv[2]}')
print(f'sys.argv[3]: {sys.argv[3]}')
```

--------------------------------

### AppTest - TextInput Interaction

Source: https://docs.streamlit.io/develop/api-reference_slug=%5B...slug%5D

Simulates inputting text into a `st.text_input` widget.

```APIDOC
## TextInput

### Description
A representation of `st.text_input`.

### Method
Python (AppTest interaction)

### Endpoint
N/A (AppTest method)

### Parameters
None

### Request Example
```python
at.text_input[0].input("Streamlit").run()
```

### Response
#### Success Response (200)
N/A (This is an interaction, not a direct response)

#### Response Example
N/A
```

--------------------------------

### Simulate Selectbox Selection (Python)

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

Simulates selecting an option from a selectbox (dropdown) element using `AppTest`. This allows testing of single-choice dropdown functionalities.

```python
at.selectbox[0].select("New York").run()
```

--------------------------------

### Create Admin Page Stub in Streamlit

Source: https://docs.streamlit.io/develop/tutorials/multipage/dynamic-navigation

This Python code snippet generates a header and a user role display for an admin page within a Streamlit application. It relies on `st.session_state.role` being previously defined. This is a template for other similar pages.

```python
import streamlit as st

st.header("Admin 1")
st.write(f"You are logged in as {st.session_state.role}.")
```

--------------------------------

### Streamlit Control Flow and Navigation

Source: https://docs.streamlit.io/develop/quick-reference/cheat-sheet

Demonstrates Streamlit functions for controlling script execution and navigation, including stopping execution (`st.stop`), rerunning the script (`st.rerun`), switching pages (`st.switch_page`), defining navigation menus (`st.navigation`), grouping widgets in forms (`st.form`), creating dialogs (`st.dialog`), and using fragments (`st.fragment`).

```python
# Stop execution immediately:
st.stop()
# Rerun script immediately:
st.rerun()
# Navigate to another page:
st.switch_page("pages/my_page.py")

# Define a navigation widget in your entrypoint file
pg = st.navigation(
    st.Page("page1.py", title="Home", url_path="home", default=True)
    st.Page("page2.py", title="Preferences", url_path="settings")
)
pg.run()

# Group multiple widgets:
with st.form(key="my_form"):
    username = st.text_input("Username")
    password = st.text_input("Password")
    st.form_submit_button("Login")

# Define a dialog function
@st.dialog("Welcome!")
def modal_dialog():
    st.write("Hello")

modal_dialog()

# Define a fragment
@st.fragment
def fragment_function():
    df = get_data()
    st.line_chart(df)
    st.button("Update")

fragment_function()
```

--------------------------------

### AppTest - Multiselect Interaction

Source: https://docs.streamlit.io/develop/api-reference_slug=%28&slug=develop&slug=concepts&slug=configuration

Simulates selecting options in the `st.multiselect` widget.

```APIDOC
## Multiselect

### Description
A representation of `st.multiselect`.

### Method
Python

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
at.multiselect[0].select("New York").run()
```

### Response
#### Success Response (200)
N/A

#### Response Example
N/A
```

--------------------------------

### Super-Admin Page Role Check in Streamlit

Source: https://docs.streamlit.io/develop/tutorials/multipage/st

This Streamlit page is exclusively for 'super-admin' roles. It employs `menu_with_redirect()` for initial authentication and then verifies if the user's role in session state is 'super-admin'. Unauthorized users receive a warning and the script execution is stopped using `st.stop()`.

```python
import streamlit as st
from menu import menu_with_redirect

# Redirect to app.py if not logged in, otherwise show the navigation menu
menu_with_redirect()

# Verify the user's role
if st.session_state.role not in ["super-admin"]:
    st.warning("You do not have permission to view this page.")
    st.stop()

st.title("This page is available to super-admins")
st.markdown(f"You are currently logged with the role of {st.session_state.role}.")
```

--------------------------------

### Test and Display Generated Data

Source: https://docs.streamlit.io/develop/tutorials/execution-flow/trigger-a-full-script-rerun-from-a-fragment

This Python snippet demonstrates how to call the `get_data` function and display the resulting DataFrame directly in a Streamlit app. This is useful for testing the data generation process and visualizing the output.

```Python
data = get_data()
data
```

--------------------------------

### Simulate Toggle State Change (Python)

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

Simulates changing the state of a toggle element (e.g., on/off) using `AppTest`. This allows testing of toggle switch functionalities.

```python
at.toggle[0].set_value("True").run()
```

--------------------------------

### Set Page Configuration Multiple Times with st.set_page_config

Source: https://docs.streamlit.io/develop/quick-reference/release-notes/2025

Streamlit now supports calling `st.set_page_config` multiple times within a single script run. This provides more flexibility in dynamically configuring page settings during script execution, overriding previous configurations as needed.

```python
import streamlit as st

st.set_page_config(page_title="Home Page", layout="wide")
# ... some code ...
st.set_page_config(page_title="Dashboard", layout="centered")
```

--------------------------------

### Tabs Layout

Source: https://docs.streamlit.io/develop/quick-reference/cheat-sheet

Organize content into different tabs for a cleaner interface.

```APIDOC
## Tabs

### Description
Create tabbed sections within your application to categorize and display content.

### Method
Python

### Endpoint
N/A

### Parameters
None

### Request Example
```python
# Insert containers separated into tabs:
tab1, tab2 = st.tabs(["Tab 1", "Tab2"])
tab1.write("this is tab 1")
tab2.write("this is tab 2")

# You can also use "with" notation:
with tab1:
    st.radio("Select one:", [1, 2])
```

### Response
N/A
```

--------------------------------

### Configure Navigation in Streamlit Multipage Apps

Source: https://docs.streamlit.io/develop/api-reference/navigation

Defines the structure and content of navigation for a multipage Streamlit application. It takes a dictionary where keys are section titles and values are lists of pages within that section. This helps organize the user interface for navigating between different parts of the app.

```python
st.navigation({
    "Your account" : [log_out, settings],
    "Reports" : [overview, usage],
    "Tools" : [search]
})
```

--------------------------------

### Streamlit: Dynamically Insert Content with st.empty (Python)

Source: https://docs.streamlit.io/knowledge-base/using-streamlit/insert-elements-out-of-order

This Python code snippet demonstrates how to use `st.empty()` to create placeholders in a Streamlit app. These placeholders can be updated later with different content, such as text or charts, allowing for dynamic content insertion and out-of-order display.

```python
import streamlit as st
import numpy as np

st.text('This will appear first')
# Appends some text to the app.

my_slot1 = st.empty()
# Appends an empty slot to the app. We'll use this later.

my_slot2 = st.empty()
# Appends another empty slot.

st.text('This will appear last')
# Appends some more text to the app.

my_slot1.text('This will appear second')
# Replaces the first empty slot with a text string.

my_slot2.line_chart(np.random.randn(20, 2))
# Replaces the second empty slot with a chart.
```

--------------------------------

### Display Monthly Sales Data in Streamlit

Source: https://docs.streamlit.io/develop/tutorials/execution-flow/trigger-a-full-script-rerun-from-a-fragment

This Python function displays monthly sales data using Streamlit. It calculates the sales for the selected month from the provided data and visualizes the total sales for that month using a bar chart. It requires the 'streamlit' and 'datetime' libraries.

```Python
import streamlit as st
import time
from datetime import timedelta

def show_monthly_sales(data):
    time.sleep(1)
    selected_date = st.session_state.selected_date
    this_month = selected_date.replace(day=1)
    next_month = (selected_date.replace(day=28) + timedelta(days=4)).replace(day=1)

    st.header(f"Daily sales for all products, {this_month:%B %Y}")
    monthly_sales = data[(data.index < next_month) & (data.index >= this_month)]
    st.write(monthly_sales)

    st.header(f"Total sales for all products, {this_month:%B %Y}")
    st.bar_chart(monthly_sales.sum())
```

--------------------------------

### Select User Role with Streamlit

Source: https://docs.streamlit.io/develop/tutorials/multipage/st

This snippet shows how to create a selectbox in Streamlit to allow users to choose their role. The selected role is stored in session state and can be used to control access to different parts of the application. It also triggers a `set_role` function and renders a dynamic menu.

```python
import streamlit as st

st.selectbox(
    "Select your role:",
    [None, "user", "admin", "super-admin"],
    key="_role",
    on_change=set_role,
)
menu() # Render the dynamic menu!
```

--------------------------------

### Serve Static Image in Streamlit App

Source: https://docs.streamlit.io/develop/concepts/configuration/serving-static-files

This Python code snippet demonstrates how to display a static image served by Streamlit. The image `cat.png` is assumed to be in the `./static/` folder and is linked in a markdown element. Ensure `enableStaticServing` is true in your Streamlit configuration.

```python
# app.py
import streamlit as st

with st.echo():
    st.title("CAT")

    st.markdown("[![Click me](app/static/cat.png)](https://streamlit.io)")
```

--------------------------------

### Building a Static Component Export

Source: https://docs.streamlit.io/develop/concepts/custom-components/intro

This command is used to build a static, production-ready version of your Streamlit custom component. This is typically done after development is complete and before deployment.

```bash
npm run export
```

--------------------------------

### Streamlit CLI: Configuration

Source: https://docs.streamlit.io/develop/api-reference/cli

Command to display all current Streamlit configuration options. This helps in understanding and verifying settings.

```bash
streamlit config show
```

--------------------------------

### Display a multi-line text input widget in Python

Source: https://docs.streamlit.io/develop/api-reference/widgets

Renders a larger text area for multi-line text input. Returns the entered string. Suitable for longer text, descriptions, or content that requires multiple lines.

```python
text = st.text_area("Text to translate")
```

--------------------------------

### Streamlit Deployment Dependencies

Source: https://docs.streamlit.io/develop/tutorials/authentication/microsoft

Specifies the Python dependencies required for deploying a Streamlit application to Streamlit Community Cloud. It lists `streamlit` and `Authlib` with version constraints to ensure compatibility.

```txt
streamlit>=1.42.0
Authlib>=1.3.2

```

--------------------------------

### Cache Global Resources with Streamlit

Source: https://docs.streamlit.io/develop/quick-reference/cheat-sheet

Explains how to use the `@st.cache_resource` decorator to cache global resources like TensorFlow sessions or database connections. This improves performance by reusing expensive-to-create objects.

```python
# E.g. TensorFlow session, database connection, etc.
@st.cache_resource
def foo(bar):
    # Create and return a non-data object
    return session
# Executes foo
s1 = foo(ref1)
# Does not execute foo
# Returns cached item by reference, s1 == s2
s2 = foo(ref1)
# Different arg, so function foo executes
s3 = foo(ref2)
# Clear the cached value for foo(ref1)
foo.clear(ref1)
# Clear all cached entries for this function
foo.clear()
# Clear all global resources from cache
st.cache_resource.clear()
```

--------------------------------

### AppTest - Selectbox Interaction

Source: https://docs.streamlit.io/develop/api-reference_slug=%28&slug=develop&slug=concepts&slug=configuration

Simulates selecting an option from the `st.selectbox` widget.

```APIDOC
## Selectbox

### Description
A representation of `st.selectbox`.

### Method
Python

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
at.selectbox[0].select("New York").run()
```

### Response
#### Success Response (200)
N/A

#### Response Example
N/A
```

--------------------------------

### Display Almost Anything

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=configuration&slug=text

This section covers methods for writing and displaying various types of content in a Streamlit app, including text, dataframes, figures, and streams.

```APIDOC
## Display Almost Anything

### st.write

**Description**: Write arguments to the app.

**Method**: Python Function

**Endpoint**: N/A

**Parameters**:

#### Arguments
- **args** (any) - The arguments to write to the app.

### st.write_stream

**Description**: Write generators or streams to the app with a typewriter effect.

**Method**: Python Function

**Endpoint**: N/A

**Parameters**:

#### Arguments
- **stream_data** (generator or stream) - The data to stream to the app.

### Magic

**Description**: Automatically writes standalone variables or literals to your app using `st.write`.

**Method**: Implicit Streamlit Feature

**Endpoint**: N/A

**Parameters**: None
```

--------------------------------

### Adding Component Packages with npm

Source: https://docs.streamlit.io/develop/concepts/custom-components/intro

This command demonstrates how to add new JavaScript packages to your Streamlit custom component's frontend dependencies. It's run from within the component's `frontend/` directory.

```bash
npm add baseui
```

--------------------------------

### Integrating Lottie Animations in Streamlit

Source: https://docs.streamlit.io/develop/api-reference_slug=publish

Adds Lottie animations to Streamlit apps by loading animation data from a URL and displaying it. Requires the streamlit-lottie library. The `key` argument is used to manage the animation's state.

```python
from streamlit_lottie import st_lottie, load_lottieurl
import streamlit as st

lottie_hello = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_V9t630.json")
st_lottie(lottie_hello, key="hello")
```

--------------------------------

### ChatInput Interaction

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=configuration&slug=layouts

Simulates user input in the `st.chat_input` widget.

```APIDOC
## ChatInput

### Description
A representation of `st.chat_input`.

### Method
Python (AppTest interaction)

### Endpoint
N/A

### Parameters
None

### Request Example
```python
at.chat_input[0].set_value("What is Streamlit?").run()
```

### Response
N/A

### Response Example
N/A
```

--------------------------------

### Accessing Base Elements

Source: https://docs.streamlit.io/develop/api-reference/app-testing

Illustrates how to access base elements like `st.title` which inherit from the `Element` class. This allows for inspecting the values and properties of various displayed elements.

```Python
# at.title returns a sequence of Title
# Title inherits from Element
assert at.title[0].value == "My awesome app"
```

--------------------------------

### Configure Streamlit Secrets for Supabase (TOML)

Source: https://docs.streamlit.io/develop/tutorials/databases/supabase

TOML configuration for Streamlit's secrets management, specifying the Supabase Project URL and API Key. This file should be placed in `.streamlit/secrets.toml` and added to `.gitignore`.

```toml
# .streamlit/secrets.toml

[connections.supabase]
SUPABASE_URL = "xxxx"
SUPABASE_KEY = "xxxx"
```

--------------------------------

### AppTest - Selectbox Interaction

Source: https://docs.streamlit.io/develop/api-reference_slug=%5B...slug%5D

Simulates selecting an option from a `st.selectbox` widget.

```APIDOC
## Selectbox

### Description
A representation of `st.selectbox`.

### Method
Python (AppTest interaction)

### Endpoint
N/A (AppTest method)

### Parameters
None

### Request Example
```python
at.selectbox[0].select("New York").run()
```

### Response
#### Success Response (200)
N/A (This is an interaction, not a direct response)

#### Response Example
N/A
```

--------------------------------

### AppTest.run

Source: https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1

Run the script from the current state, simulating a rerun or user interaction.

```APIDOC
## AppTest.run

### Description
Run the script from the current state. This is equivalent to manually rerunning the app or the rerun that occurs upon user interaction. `AppTest.run()` must be manually called after updating a widget value or switching pages as script reruns do not occur automatically as they do for live-running Streamlit apps.

### Method
POST

### Endpoint
/websites/streamlit_io/AppTest/run

### Parameters
#### Query Parameters
- **timeout** (float or None) - Optional - The maximum number of seconds to run the script. If `timeout` is `None` (default), Streamlit uses the default timeout set for the instance of `AppTest`.

### Response
#### Success Response (200)
- **AppTest Instance** (AppTest) - self

#### Response Example
```json
{
  "message": "AppTest script ran successfully"
}
```
```

--------------------------------

### Add s3fs and st-files-connection to requirements.txt

Source: https://docs.streamlit.io/develop/tutorials/databases/aws-s3

Lists the necessary Python packages for Streamlit to interact with AWS S3. It's recommended to pin the versions for reproducibility.

```bash
# requirements.txt
s3fs==x.x.x
st-files-connection
```

--------------------------------

### Initialize Snowflake Connection in Python

Source: https://docs.streamlit.io/develop/api-reference/connections

Initializes a Streamlit connection object specifically for Snowflake. This allows interaction with Snowflake data warehouses.

```python
conn = st.connection('snowflake')
```

--------------------------------

### Simulate Chat Response Stream (Python)

Source: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/chat-response-feedback

This Python function simulates a streaming chat response by yielding characters of a predefined response string with a small delay between each character. This is useful for creating a more interactive chat experience in Streamlit applications.

```Python
def chat_stream(prompt):
    response = f'You said, "{prompt}" ...interesting.'
    for char in response:
        yield char
        time.sleep(0.02)
```

--------------------------------

### Streamlit App to Query TiDB (Python)

Source: https://docs.streamlit.io/develop/tutorials/databases/tidb

A Python script for a Streamlit application that establishes a connection to a TiDB database using `st.connection` and executes a SQL query to fetch data. The results are then displayed in the Streamlit app. Caching is configured with a TTL of 600 seconds.

```python
# streamlit_app.py

import streamlit as st

# Initialize connection.
conn = st.connection('tidb', type='sql')

# Perform query.
df = conn.query('SELECT * from mytable;', ttl=600)

# Print results.
for row in df.itertuples():
    st.write(f"{row.name} has a :{row.pet}:")
```

--------------------------------

### Display a selectbox widget in Python

Source: https://docs.streamlit.io/develop/api-reference/widgets

Renders a dropdown selectbox where users can choose a single option from a list. Returns the selected option. Similar to radio buttons but more compact for longer lists.

```python
choice = st.selectbox("Pick one", ["cats", "dogs"])
```

--------------------------------

### Custom Components API

Source: https://docs.streamlit.io/develop/api-reference

Documentation for creating and using custom components in Streamlit.

```APIDOC
## Declare Component

### Description
Create and register a custom component.

### Method
Python

### Endpoint
`st.components.v1.declare_component`

### Parameters
#### Path Parameters
- **name** (str) - Required - The name of the component.
- **path** (str) - Required - The path to the component's frontend directory.

### Request Example
```python
from st.components.v1 import declare_component
declare_component(
    "custom_slider",
    "/frontend",
)
```

## HTML Component

### Description
Display an HTML string in an iframe.

### Method
Python

### Endpoint
`st.components.v1.html`

### Parameters
#### Path Parameters
- **content** (str) - Required - The HTML content to display.

### Request Example
```python
from st.components.v1 import html
html(
    "<p>Foo bar.</p>"
)
```

## Iframe Component

### Description
Load a remote URL in an iframe.

### Method
Python

### Endpoint
`st.components.v1.iframe`

### Parameters
#### Path Parameters
- **url** (str) - Required - The URL to load in the iframe.

### Request Example
```python
from st.components.v1 import iframe
iframe(
    "docs.streamlit.io"
)
```
```

--------------------------------

### AppTest - Toggle Interaction

Source: https://docs.streamlit.io/develop/api-reference_slug=%28&slug=develop&slug=concepts&slug=configuration

Simulates setting the state of the `st.toggle` widget.

```APIDOC
## Toggle

### Description
A representation of `st.toggle`.

### Method
Python

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
at.toggle[0].set_value("True").run()
```

### Response
#### Success Response (200)
N/A

#### Response Example
N/A
```

--------------------------------

### Implement Google Login Screen in Streamlit

Source: https://docs.streamlit.io/develop/tutorials/authentication/google

This Python function defines the UI for a private Streamlit application's login screen. It displays a header, a subheader, and a 'Log in with Google' button. The button's `on_click` event is set to `st.login`, which initiates the Google OAuth flow when clicked.

```python
def login_screen():
    st.header("This app is private.")
    st.subheader("Please log in.")
    st.button("Log in with Google", on_click=st.login)
```

--------------------------------

### Live Camera Input for Streamlit

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

Provides a live webcam image feed as an alternative to st.camera_input. It captures and displays the webcam feed directly in the Streamlit app. Requires the 'camera-input-live' library.

```python
from camera_input_live import camera_input_live

image = camera_input_live()
st.image(value)
```

--------------------------------

### Streamlit App to Connect and Fetch Tableau Data

Source: https://docs.streamlit.io/develop/tutorials/databases/tableau

This Python script demonstrates how to connect to Tableau Server using `tableauserverclient` and Streamlit's secrets management. It fetches workbooks, views, images, and CSV data, utilizing `st.cache_data` to optimize performance by caching query results for a specified duration.

```python
# streamlit_app.py

import streamlit as st
import tableauserverclient as TSC
import pandas as pd
from io import StringIO


# Set up connection.
tableau_auth = TSC.PersonalAccessTokenAuth(
    st.secrets["tableau"]["token_name"],
    st.secrets["tableau"]["personal_access_token"],
    st.secrets["tableau"]["site_id"],
)
server = TSC.Server(st.secrets["tableau"]["server_url"], use_server_version=True)


# Get various data.
# Explore the tableauserverclient library for more options.
# Uses st.cache_data to only rerun when the query changes or after 10 min.
@st.cache_data(ttl=600)
def run_query():
    with server.auth.sign_in(tableau_auth):

        # Get all workbooks.
        workbooks, pagination_item = server.workbooks.get()
        workbooks_names = [w.name for w in workbooks]

        # Get views for first workbook.
        server.workbooks.populate_views(workbooks[0])
        views_names = [v.name for v in workbooks[0].views]

        # Get image & CSV for first view of first workbook.
        view_item = workbooks[0].views[0]
        server.views.populate_image(view_item)
        server.views.populate_csv(view_item)
        view_name = view_item.name
        view_image = view_item.image
        # `view_item.csv` is a list of binary objects, convert to str.
        view_csv = b"".join(view_item.csv).decode("utf-8")

        return workbooks_names, views_names, view_name, view_image, view_csv

workbooks_names, views_names, view_name, view_image, view_csv = run_query()


# Print results.
st.subheader("📓 Workbooks")
st.write("Found the following workbooks:", ", ".join(workbooks_names))

st.subheader("👁️ Views")
st.write(
    f"Workbook *{workbooks_names[0]}* has the following views:",
    ", ".join(views_names),
)

st.subheader("🖼️ Image")
st.write(f"Here's what view *{view_name}* looks like:")
st.image(view_image, width=300)

st.subheader("📊 Data")
st.write(f"And here's the data for view *{view_name}*:")
st.write(pd.read_csv(StringIO(view_csv)))

```

--------------------------------

### Filter and Plot Map Data by Hour in Streamlit

Source: https://docs.streamlit.io/get-started/tutorials/create-an-app

Filters the dataset to show only pickups occurring at a specific hour and then plots this filtered data on a map. This allows for time-based analysis of pickup concentrations.

```python
hour_to_filter = 17
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
st.subheader(f'Map of all pickups at {hour_to_filter}:00')
st.map(filtered_data)
```

--------------------------------

### Display a simple toast message with Streamlit st.toast

Source: https://docs.streamlit.io/develop/api-reference/status/st

Demonstrates how to display a basic toast message with a custom icon using Streamlit's st.toast function. This is useful for providing quick feedback to the user.

```python
import streamlit as st

st.toast("Your edited image was saved!", icon="😍")
```

--------------------------------

### Configuration API

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=configuration&slug=charts

APIs for managing Streamlit application configuration, including setting the page title, favicon, and retrieving/setting specific options.

```APIDOC
## Configuration file

### Description
Configures the default settings for your app using a TOML file.

### Method
File Structure

### Endpoint
`your-project/.streamlit/config.toml`

### Request Example
```toml
[theme]
primaryColor="#FF4B4B"
```

## Get config option

### Description
Retrieve a single configuration option.

### Method
Python

### Endpoint
`st.get_option`

### Parameters
#### Path Parameters
- **key** (str) - Required - The configuration option key (e.g., "theme.primaryColor").

### Request Example
```python
st.get_option("theme.primaryColor")
```

## Set config option

### Description
Set a single configuration option. (This is very limited.)

### Method
Python

### Endpoint
`st.set_option`

### Parameters
#### Path Parameters
- **key** (str) - Required - The configuration option key.
- **value** - Required - The value to set for the option.

### Request Example
```python
st.set_option("deprecation.showPyplotGlobalUse", False)
```

## Set page title, favicon, and more

### Description
Configures the default settings of the page, such as title and icon.

### Method
Python

### Endpoint
`st.set_page_config`

### Parameters
#### Path Parameters
- **page_title** (str) - Optional - The title of the app page.
- **page_icon** (str) - Optional - The favicon for the app page (e.g., ":shark:").
- **layout** (str) - Optional - The layout of the app ('centered' or 'wide').
- **initial_sidebar_state** (str) - Optional - The initial state of the sidebar ('auto', 'expanded', or 'collapsed').
- **menu_items** (dict) - Optional - Custom menu items for the app.

### Request Example
```python
st.set_page_config(
  page_title="My app",
  page_icon=":shark:",
)
```
```

--------------------------------

### TypeScript Component Initialization and Rendering

Source: https://docs.streamlit.io/develop/concepts/custom-components/intro

This TypeScript code demonstrates the core mechanics for a Streamlit custom component. It handles component readiness, subscribes to render events, accesses arguments, sends data back to Python, and updates frame height.

```typescript
Streamlit.setComponentReady();
Streamlit.RENDER_EVENT.addListener(onRender);

function onRender(event) {
  // Access arguments: event.detail.args
  // Send data back: Streamlit.setComponentValue(data)
  // Update frame height: Streamlit.setFrameHeight()
}
```

--------------------------------

### Configure DataFrame Border Color with theme.dataframeBorderColor

Source: https://docs.streamlit.io/develop/quick-reference/release-notes/2025

Enables setting a specific border color for dataframes and tables, distinct from other border colors in the Streamlit theme. This is configured via the `theme.dataframeBorderColor` option in `config.toml`. It allows for more granular control over the visual styling of dataframes.

```toml
[theme]
dataframeBorderColor = "#FF0000"
```

--------------------------------

### Simulate Date Input in Python

Source: https://docs.streamlit.io/develop/api-reference

Simulates setting a date value for a date input element in a Streamlit app using `AppTest`. This is useful for testing date selections.

```python
from datetime import date

release_date = date(2023, 10, 26)
at.date_input[0].set_value(release_date).run()
```

--------------------------------

### Initialize AppTest from String

Source: https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1

Initializes an AppTest instance from a string containing Streamlit script code. This is useful for testing small snippets or dynamically generated scripts. It takes the script content as a string argument.

```python
app = st.testing.v1.AppTest.from_string("import streamlit as st\nst.write('Hello, world!')")
```

--------------------------------

### Simulate Text Input (Python)

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

Simulates entering text into a text input field using `AppTest`. This is used for testing single-line text input functionalities.

```python
at.text_input[0].input("Streamlit").run()
```

--------------------------------

### Display Formatted Text with st.markdown in Python

Source: https://docs.streamlit.io/develop/api-reference/text/st

This snippet shows how to use st.markdown to display bold, italic, and colored text in Streamlit. It also demonstrates how to render emojis and handle multi-line strings with soft and hard returns.

```python
import streamlit as st

st.markdown("*Streamlit* is **really** ***cool***.")
st.markdown('''
    :red[Streamlit] :orange[can] :green[write] :blue[text] :violet[in]
    :gray[pretty] :rainbow[colors] and :blue-background[highlight] text.
''')
st.markdown("Here's a bouquet &mdash;            :tulip::cherry_blossom::rose::hibiscus::sunflower::blossom:")

multi = '''If you end a line with two spaces,
a soft return is used for the next line.

Two (or more) newline characters in a row will result in a hard return.
'''
st.markdown(multi)
```

--------------------------------

### Plost Simple Plotting Library

Source: https://docs.streamlit.io/develop/api-reference_slug=%28&slug=develop&slug=concepts&slug=configuration

Utilizes the Plost library for creating simple charts like line charts within Streamlit. Requires plost and pandas.

```Python
import plost
import streamlit as st
import pandas as pd

# Assuming my_dataframe is a pandas DataFrame
# Example:
# my_dataframe = pd.DataFrame({
#     'time': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
#     'stock_value': [100, 110, 105],
#     'stock_name': ['AAPL', 'AAPL', 'AAPL']
# })

plost.line_chart(my_dataframe, x='time', y='stock_value', color='stock_name')
```

--------------------------------

### Streamlit Chat App with Feedback Collection (Python)

Source: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/chat-response-feedback

This Python code implements a Streamlit chat application that allows users to input messages and receive echoed responses. It integrates st.feedback to collect 'thumbs up' or 'thumbs down' sentiment for each assistant response. The feedback is stored in session state and displayed alongside the chat messages.

```Python
import streamlit as st
import time

def chat_stream(prompt):
    response = f'You said, "{prompt}" ...interesting.'
    for char in response:
        yield char
        time.sleep(0.02)

def save_feedback(index):
    st.session_state.history[index]["feedback"] = st.session_state[f"feedback_{index}"]

if "history" not in st.session_state:
    st.session_state.history = []

for i, message in enumerate(st.session_state.history):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            feedback = message.get("feedback", None)
            st.session_state[f"feedback_{i}"] = feedback
            st.feedback(
                "thumbs",
                key=f"feedback_{i}",
                disabled=feedback is not None,
                on_change=save_feedback,
                args=[i],
            )

if prompt := st.chat_input("Say something"):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        response = st.write_stream(chat_stream(prompt))
        st.feedback(
            "thumbs",
            key=f"feedback_{len(st.session_state.history)}",
            on_change=save_feedback,
            args=[len(st.session_state.history)],
        )
    st.session_state.history.append({"role": "assistant", "content": response})
```

--------------------------------

### Streamlit Secrets Configuration for Google OAuth

Source: https://docs.streamlit.io/develop/tutorials/authentication/google

This TOML file contains the necessary configuration for Streamlit to connect with the Google OAuth service. It includes the redirect URI, a secret for cookie encryption, and the client ID and secret obtained from the Google Cloud Console, along with the server metadata URL for OpenID Connect.

```toml
[auth]
redirect_uri = "http://localhost:8501/oauth2callback"
cookie_secret = "xxx"
client_id = "xxx"
client_secret = "xxx"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

```

--------------------------------

### Create Annotations DataFrame using Pandas

Source: https://docs.streamlit.io/develop/tutorials/elements/annotate-an-altair-chart

Prepares annotation data by creating a Pandas DataFrame. Each row contains a date, price, marker (emoji), and description for specific points of interest on the chart. Dates are converted to datetime objects.

```Python
ANNOTATIONS = [
    ("Sep 01, 2007", 450, "🙂", "Something's going well for GOOG & AAPL."),
    ("Nov 01, 2008", 220, "🙂", "The market is recovering."),
    ("Dec 01, 2007", 750, "😱", "Something's going wrong for GOOG & AAPL."),
    ("Dec 01, 2009", 680, "😱", "A hiccup for GOOG."),
] Лannotations_df = pd.DataFrame(
    ANNOTATIONS, columns=["date", "price", "marker", "description"]
) Лannotations_df.date = pd.to_datetime(Лannotations_df.date)
```

--------------------------------

### Streamlit Connection with TTL

Source: https://docs.streamlit.io/develop/concepts/connections/connecting-to-data

Demonstrates how to configure a Streamlit connection with a Time-To-Live (TTL) for automatic expiration. This is useful for connections where data might become stale over time.

```python
st.connection('myconn', type=MyConnection, ttl=<N>)
```

--------------------------------

### AppTest - DateInput Interaction

Source: https://docs.streamlit.io/develop/api-reference_slug=%28&slug=develop&slug=concepts&slug=configuration

Simulates setting a date value for the `st.date_input` widget.

```APIDOC
## DateInput

### Description
A representation of `st.date_input`.

### Method
Python

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
from datetime import datetime
release_date = datetime.date(2023, 10, 26)
at.date_input[0].set_value(release_date).run()
```

### Response
#### Success Response (200)
N/A

#### Response Example
N/A
```

--------------------------------

### Shared Module State Across Streamlit Pages

Source: https://docs.streamlit.io/develop/concepts/multipage-apps/pages-directory

Demonstrates how Python modules are shared globally across different pages in a Streamlit multipage app. Changes made to a module in one page are reflected in other pages that import the same module, illustrating shared state.

```python
# page1.py
import foo
foo.hello = 123

# page2.py
import foo
import streamlit as st
st.write(foo.hello)  # If page1 already executed, this writes 123
```

--------------------------------

### Initialize AppTest from Function (Python)

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

Initializes `st.testing.v1.AppTest` from a Python callable function that represents a Streamlit app. This allows for testing apps defined as functions.

```python
from streamlit.testing.v1 import AppTest

at = AppTest.from_function(app_script_as_callable)
at.run()
```

--------------------------------

### Create a Multi-Element Container in Streamlit

Source: https://docs.streamlit.io/develop/api-reference/layout

This code shows how to create a container in Streamlit to group multiple elements together. Content written to the container will appear before content written outside of it.

```python
c = st.container()
st.write("This will show last")
c.write("This will show first")
c.write("This will show second")
```

--------------------------------

### Simulate Text Area Input (Python)

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

Simulates entering text into a text area element using `AppTest`. This allows testing of multi-line text input fields.

```python
at.text_area[0].input("Streamlit is awesome!").run()
```

--------------------------------

### Simulate Multiselect Selection (Python)

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

Simulates selecting an option from a multiselect element using `AppTest`. This allows testing of functionalities involving multiple selections.

```python
at.multiselect[0].select("New York").run()
```

--------------------------------

### ChatInput Testing

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

Simulates interactions with the `st.chat_input` element.

```APIDOC
## ChatInput Testing

### Description
A representation of `st.chat_input`.

### Method
Python

### Endpoint
N/A

### Parameters
#### Path Parameters
N/A

#### Query Parameters
N/A

#### Request Body
N/A

### Request Example
```python
at.chat_input[0].set_value("What is Streamlit?").run()
```

### Response
#### Success Response (200)
N/A

#### Response Example
N/A
```

--------------------------------

### Modify Feedback Widget Behavior in Streamlit

Source: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/chat-response-feedback

Demonstrates how to modify the behavior of the `st.feedback` widget in Streamlit. This includes options to allow rating only the most recent response or to enable changing previously submitted ratings.

```Python
for i, message in enumerate(st.session_state.history):
      with st.chat_message(message["role"):
          st.write(message["content")
-         if message["role"] == "assistant":
-             feedback = message.get("feedback", None)
-             st.session_state[f"feedback_{i}"] = feedback
-             st.feedback(
-                 "thumbs",
-                 key=f"feedback_{i}",
-                 disabled=feedback is not None,
-                 on_change=save_feedback,
-                 args=[i],
-             )
```

```Python
for i, message in enumerate(st.session_state.history):
      with st.chat_message(message["role"):
          st.write(message["content")
          if message["role"] == "assistant":
              feedback = message.get("feedback", None)
              st.session_state[f"feedback_{i}"] = feedback
              st.feedback(
                  "thumbs",
                  key=f"feedback_{i}",
-                 disabled=feedback is not None,
                  on_change=save_feedback,
                  args=[i],
              )
```

--------------------------------

### Run Streamlit App (Subdirectory)

Source: https://docs.streamlit.io/develop/api-reference/cli/run

Launches a Streamlit app located within a subdirectory. Streamlit will look for `streamlit_app.py` inside the specified directory.

```bash
streamlit run your_subdirectory
```

--------------------------------

### AppTest - Radio Interaction

Source: https://docs.streamlit.io/develop/api-reference_slug=%28&slug=develop&slug=concepts&slug=configuration

Simulates selecting an option in the `st.radio` widget.

```APIDOC
## Radio

### Description
A representation of `st.radio`.

### Method
Python

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
at.radio[0].set_value("New York").run()
```

### Response
#### Success Response (200)
N/A

#### Response Example
N/A
```

--------------------------------

### AppTest - Button Interaction

Source: https://docs.streamlit.io/develop/api-reference_slug=%28&slug=develop&slug=concepts&slug=configuration

Simulates user interaction with buttons, including `st.button` and `st.form_submit_button`.

```APIDOC
## Button

### Description
A representation of `st.button` and `st.form_submit_button`.

### Method
Python

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
at.button[0].click().run()
```

### Response
#### Success Response (200)
N/A

#### Response Example
N/A
```

--------------------------------

### AppTest - Button Interaction

Source: https://docs.streamlit.io/develop/api-reference_slug=%5B...slug%5D

Simulates user interaction with buttons, including `st.button` and `st.form_submit_button`.

```APIDOC
## Button

### Description
A representation of `st.button` and `st.form_submit_button`.

### Method
Python (AppTest interaction)

### Endpoint
N/A (AppTest method)

### Parameters
None

### Request Example
```python
at.button[0].click().run()
```

### Response
#### Success Response (200)
N/A (This is an interaction, not a direct response)

#### Response Example
N/A
```

--------------------------------

### Create Altair Annotation Layer

Source: https://docs.streamlit.io/develop/tutorials/elements/annotate-an-altair-chart

Generates an annotation layer using Altair, displaying text markers (emojis) at specified dates and prices. Includes tooltips that show the description associated with each annotation. The text is offset for better positioning.

```Python
annotation_layer = (
    alt.Chart(annotations_df)
    .mark_text(size=20, dx=-10, dy=0, align="left")
    .encode(x="date:T", y=alt.Y("price:Q"), text="marker", tooltip="description")
)
```

--------------------------------

### Import Streamlit Component in Python

Source: https://docs.streamlit.io/develop/concepts/custom-components

This Python code snippet demonstrates how to import the 'AgGrid' component from the 'st_aggrid' library. This import statement is necessary before you can use the component in your Streamlit application.

```python
from st_aggrid import AgGrid
```

--------------------------------

### DateInput Testing

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

Simulates interactions with the `st.date_input` element.

```APIDOC
## DateInput Testing

### Description
A representation of `st.date_input`.

### Method
Python

### Endpoint
N/A

### Parameters
#### Path Parameters
N/A

#### Query Parameters
N/A

#### Request Body
N/A

### Request Example
```python
from datetime import datetime
release_date = datetime.date(2023, 10, 26)
at.date_input[0].set_value(release_date).run()
```

### Response
#### Success Response (200)
N/A

#### Response Example
N/A
```

--------------------------------

### Accessing App Elements via Block

Source: https://docs.streamlit.io/develop/api-reference/app-testing

Demonstrates accessing elements within container blocks like `st.sidebar`. This allows for testing components nested within these containers, such as buttons.

```Python
# at.sidebar returns a Block
at.sidebar.button[0].click().run()
assert not at.exception
```

--------------------------------

### Display a checkbox widget in Python

Source: https://docs.streamlit.io/develop/api-reference/widgets

Renders a checkbox that users can select or deselect. Returns True if checked, False otherwise. Useful for boolean options or confirmations.

```python
selected = st.checkbox("I agree")
```

--------------------------------

### Display Localhost Address in Headless Streamlit Mode

Source: https://docs.streamlit.io/develop/quick-reference/release-notes/2024

Fixes an issue where `streamlit run` did not display the `localhost` address when initializing Streamlit with `server.headless=true`. This ensures users can access the running app in headless mode.

```python
# To run in headless mode:
# streamlit run your_app.py --server.headless true

# The output in the terminal should now correctly show the localhost address.
```

--------------------------------

### Simulate Color Picker Interaction (Python)

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

Simulates picking a color from a color picker element using `AppTest`. This allows testing of color selection functionalities.

```python
at.color_picker[0].pick("#FF4B4B").run()
```

--------------------------------

### Streamlit Fragment: State Update Comparison

Source: https://docs.streamlit.io/develop/api-reference/execution-flow/st

Compares how elements inside and outside a fragment update during app or fragment reruns. Clicking 'Rerun fragment' only increments the fragment's counter, while 'Rerun full app' increments both counters and updates all displayed values.

```python
import streamlit as st

if "app_runs" not in st.session_state:
    st.session_state.app_runs = 0
    st.session_state.fragment_runs = 0

@st.fragment
def my_fragment():
    st.session_state.fragment_runs += 1
    st.button("Rerun fragment")
    st.write(f"Fragment says it ran {st.session_state.fragment_runs} times.")

st.session_state.app_runs += 1
my_fragment()
st.button("Rerun full app")
st.write(f"Full app says it ran {st.session_state.app_runs} times.")
st.write(f"Full app sees that fragment ran {st.session_state.fragment_runs} times.")
```

--------------------------------

### Streamlit Camera Input Live for Real-time Images

Source: https://docs.streamlit.io/develop/api-reference_slug=%28&slug=develop&slug=concepts&slug=configuration

An alternative to `st.camera_input` that captures webcam images live. It provides a continuous stream of images from the camera, suitable for real-time applications.

```python
from camera_input_live import camera_input_live
import streamlit as st

image = camera_input_live()
if image is not None:
    st.image(image, caption='Live Camera Feed')

```

--------------------------------

### Define and Use FrontendRenderer in TypeScript

Source: https://docs.streamlit.io/develop/api-reference/custom-components/component-v2-lib-frontendrendererargs

Demonstrates defining custom state and data interfaces and using them with FrontendRenderer. It shows how to destructure arguments, query DOM elements, and use setStateValue and setTriggerValue for state management and event handling within a Streamlit custom component.

```typescript
import { FrontendRenderer, FrontendState } from '@streamlit/component-v2-lib';

interface MyFrontendState extends FrontendState {
    selected_item: string | null
    button_clicked: boolean
}

interface MyFrontendData {
    label: string
    options: string[]
}

const MyFrontendRenderer: FrontendRenderer<MyFrontendState, MyFrontendData> = (component) => {
    // Destructure the component args for easier access
    const { data, setStateValue, setTriggerValue, parentElement } = component

    // Set up event handlers with type-safe state management
    const dropdown = parentElement.querySelector('#dropdown') as HTMLSelectElement
    const button = parentElement.querySelector('#submit') as HTMLButtonElement

    dropdown.onchange = () => {
        setStateValue('selected_item', dropdown.value)
    }

    button.onclick = () => {
        setTriggerValue('button_clicked', true)
    }
}

export default MyFrontendRenderer;
```

--------------------------------

### Display a data editor widget in Python

Source: https://docs.streamlit.io/develop/api-reference/widgets

Renders an interactive table for viewing and editing dataframes. Supports dynamic row addition and editing. Returns the modified dataframe. Requires a pandas DataFrame as input.

```python
edited = st.data_editor(df, num_rows="dynamic")
```

--------------------------------

### Plost Simple Line Chart

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=widget-semantics

Generates a simple line chart using the Plost library for Streamlit. Requires a pandas DataFrame with time and stock data.

```python
import streamlit as st
import plost

# Assuming my_dataframe is a pandas DataFrame with 'time', 'stock_value', and 'stock_name' columns
# plost.line_chart(my_dataframe, x='time', y='stock_value', color='stock_name')
```

--------------------------------

### Configure Supabase Secrets for Streamlit

Source: https://docs.streamlit.io/develop/tutorials/databases/supabase

This snippet shows how to add Supabase project URL and API key to Streamlit's secrets management file (`.streamlit/secrets.toml`). This is crucial for securely accessing Supabase credentials within your Streamlit application without hardcoding them.

```toml
# .streamlit/secrets.toml

SUPABASE_URL = "xxxx"
SUPABASE_KEY = "xxxx"
```

--------------------------------

### Streamlit Extras: Mentions and Metric Cards

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

Provides useful extras for Streamlit applications, including the `mention` component for creating links with icons and labels, and `style_metric_cards` for styling metric displays.

```python
mention(label="An awesome Streamlit App", icon="streamlit",  url="https://extras.streamlit.app")
```

```python
from streamlit_extras.metric_cards import style_metric_cards
col3.metric(label="No Change", value=5000, delta=0)

style_metric_cards()
```

--------------------------------

### Streamlit st.time_input Basic Usage

Source: https://docs.streamlit.io/develop/api-reference/widgets/st

Demonstrates the basic usage of st.time_input to set a time alarm. It imports the necessary libraries, creates a time input widget with a label and a default time, and then displays the selected time.

```python
import datetime
import streamlit as st

t = st.time_input("Set an alarm for", datetime.time(8, 45))
st.write("Alarm is set for", t)
```

--------------------------------

### Display a toggle widget in Python

Source: https://docs.streamlit.io/develop/api-reference/widgets

A switch-like widget that allows users to toggle a setting on or off. Returns True when activated, False when deactivated. Ideal for boolean settings.

```python
activated = st.toggle("Activate")
```

--------------------------------

### Display a time input widget in Python

Source: https://docs.streamlit.io/develop/api-reference/widgets

Enables users to input a specific time. Returns a `datetime.time` object. Useful for setting appointment times or time-based preferences.

```python
time = st.time_input("Meeting time")
```

--------------------------------

### Displaying Headers with st.header

Source: https://docs.streamlit.io/develop/api-reference_slug=private-gsheet

Format text as a header using the `st.header` function.

```APIDOC
## POST /st.header

### Description
Display text in header formatting.

### Method
POST

### Endpoint
/st.header

### Parameters
#### Request Body
- **header_text** (string) - Required - The text to be displayed as a header.

### Request Example
```json
{
  "header_text": "This is a header"
}
```

### Response
#### Success Response (200)
- **status** (string) - Indicates successful rendering of the header.

#### Response Example
```json
{
  "status": "success"
}
```
```

--------------------------------

### Set Streamlit Configuration Option

Source: https://docs.streamlit.io/develop/api-reference/configuration

Shows how to set a single configuration option in Streamlit. This functionality is limited and primarily used for specific adjustments, such as controlling deprecation warnings.

```python
st.set_option("deprecation.showPyplotGlobalUse", False)
```

--------------------------------

### Streamlit App to Query Snowflake Data

Source: https://docs.streamlit.io/develop/tutorials/databases/snowflake

A Python code snippet for a Streamlit application that establishes a Snowflake connection using `st.connection` and executes a SQL query. It demonstrates how to retrieve data and display it, with options for query caching using the `ttl` parameter.

```python
# streamlit_app.py

import streamlit as st

conn = st.connection("snowflake")
df = conn.query("SELECT * FROM mytable;", ttl="10m")

for row in df.itertuples():
    st.write(f"{row.NAME} has a :{row.PET}:")
```

--------------------------------

### Streamlit Magic Commands: Displaying Markdown, DataFrames, and Charts in Python

Source: https://docs.streamlit.io/develop/api-reference/write-magic/magic

Demonstrates how Streamlit's Magic commands automatically render various content types directly from Python code. This includes markdown strings, pandas DataFrames, and Matplotlib charts. Magic commands are enabled by default but can be turned off.

```python
# Draw a title and some text to the app:
'''
# This is the document title

This is some _markdown_.
'''

import pandas as pd
df = pd.DataFrame({'col1': [1,2,3]})
df  # 👇 Draw the dataframe

x = 10
'x', x  # 👇 Draw the string 'x' and then the value of x

# Also works with most supported chart types
import matplotlib.pyplot as plt
import numpy as np

arr = np.random.normal(1, 1, size=100)
fig, ax = plt.subplots()
ax.hist(arr, bins=20)

fig  # 👇 Draw a Matplotlib chart

```

--------------------------------

### Create columns for action buttons in Streamlit

Source: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/validate-and-edit-chat-responses

This Python code creates three equal-width columns using `st.columns(3)`. These columns are used to arrange action buttons horizontally within the Streamlit interface, providing a structured layout for user interaction in the 'validate' stage.

```python
cols = st.columns(3)
```

--------------------------------

### Streamlit App with Built-in and External Dependencies

Source: https://docs.streamlit.io/deploy/concepts/dependencies

This Python script demonstrates a basic Streamlit app that imports Streamlit itself, along with pandas and numpy. It also imports built-in Python modules like math and random. When deploying, only Streamlit, pandas, and numpy need to be explicitly listed as dependencies if they are not implicitly handled.

```python
import streamlit as st
import pandas as pd
import numpy as np
import math
import random

st.write('Hi!')
```

--------------------------------

### Streamlit Lottie Animation

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

Integrates Lottie animations into Streamlit applications. Requires the 'streamlit-lottie' library and a URL to a Lottie JSON file.

```python
import streamlit as st
from streamlit_lottie import st_lottie, load_lottieurl

lottie_hello = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_V9t630.json")
st_lottie(lottie_hello, key="hello")
```

--------------------------------

### Combining Streamlit Embed Options

Source: https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app/embed-your-app

Illustrates how to combine multiple `?embed_options` query parameters in JavaScript to customize the embedding behavior of a Streamlit app simultaneously. This allows for a more tailored embedding experience by enabling several features at once.

```javascript
/?embed=true&embed_options=show_toolbar&embed_options=show_padding&embed_options=show_footer&embed_options=show_colored_line&embed_options=disable_scrolling
```

--------------------------------

### Personalize Streamlit Apps with User Authentication and Context

Source: https://docs.streamlit.io/develop/quick-reference/cheat-sheet

Covers features for personalizing Streamlit applications, including user authentication with `st.login()` and `st.logout()`, and accessing user context information like cookies, headers, locale, and theme settings.

```python
# Authenticate users
if not st.user.is_logged_in:
    st.login("my_provider")
f"Hi, {st.user.name}"
st.logout()

# Get dictionaries of cookies, headers, locale, and browser data
st.context.cookies
st.context.headers
st.context.ip_address
st.context.is_embedded
st.context.locale
st.context.theme.type
st.context.timezone
st.context.timezone_offset
st.context.url
```

--------------------------------

### Magic Commands

Source: https://docs.streamlit.io/develop/quick-reference/cheat-sheet

Streamlit's magic commands allow for implicit display of objects by simply writing them on a new line.

```APIDOC
## Magic Commands

### Description
Magic commands implicitly call `st.write()` for displaying various objects.

### Method
Python

### Endpoint
N/A

### Parameters
None

### Request Example
```python
# Magic commands implicitly call st.write().
"_This_ is some **Markdown**"
my_variable
"dataframe:", my_data_frame
```

### Response
N/A
```

--------------------------------

### Configuring Streamlit Options via Command Line or Config File

Source: https://docs.streamlit.io/develop/quick-reference/release-notes/2019

Introduces the ability to set Streamlit configuration options using command-line flags or a local configuration file. This offers multiple ways to customize Streamlit's behavior.

```bash
streamlit run app.py --server.port 8501
```

```ini
[server]
port = 8501

```

--------------------------------

### Third-Party Components

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=configuration&slug=layouts

Integrate community-developed components like Stqdm, custom notification boxes, and Streamlit Extras for enhanced functionality.

```APIDOC
## Third-Party Components

### Description
Integrate community-developed components like Stqdm, custom notification boxes, and Streamlit Extras for enhanced functionality.

### Stqdm
Provides a simple way to handle progress bars in Streamlit apps.

```python
from stqdm import stqdm

for _ in stqdm(range(50)):
    sleep(0.5)
```

### Custom Notification Box
A customizable notification box with a close option.

```python
from streamlit_custom_notification_box import custom_notification_box

styles = {
    'material-icons': {'color': 'red'},
    'text-icon-link-close-container': {'box-shadow': '#3896de 0px 4px'},
    'notification-text': {'': ''},
    'close-button': {'': ''},
    'link': {'': ''}
}
custom_notification_box(
    icon='info',
    textDisplay='We are almost done with your registration...', 
    externalLink='more info',
    url='#',
    styles=styles,
    key="foo"
)
```

### Streamlit Extras
A library offering various useful extras for Streamlit applications.

```python
from streamlit_extras.let_it_rain import rain

rain(
    emoji="🎈", 
    font_size=54, 
    falling_speed=5,
    animation_length="infinite",
)
```
```

--------------------------------

### Display Content in the Streamlit Sidebar

Source: https://docs.streamlit.io/develop/api-reference/layout

This code illustrates how to place elements in the Streamlit sidebar using `st.sidebar`. Any Streamlit element can be written to the sidebar.

```python
st.sidebar.write("This lives in the sidebar")
st.sidebar.button("Click me!")
```

--------------------------------

### Streamlit Magic Commands for Implicit st.write()

Source: https://docs.streamlit.io/develop/quick-reference/cheat-sheet

Illustrates Streamlit's 'magic commands' which implicitly call `st.write()`. This allows for direct display of Markdown, variables, and dataframes without explicitly calling `st.write()`.

```python
# Magic commands implicitly
# call st.write().
"_This_ is some **Markdown**"
my_variable
"dataframe:", my_data_frame
```

--------------------------------

### AppTest - TextArea Interaction

Source: https://docs.streamlit.io/develop/api-reference_slug=%5B...slug%5D

Simulates inputting text into a `st.text_area` widget.

```APIDOC
## TextArea

### Description
A representation of `st.text_area`.

### Method
Python (AppTest interaction)

### Endpoint
N/A (AppTest method)

### Parameters
None

### Request Example
```python
at.text_area[0].input("Streamlit is awesome!").run()
```

### Response
#### Success Response (200)
N/A (This is an interaction, not a direct response)

#### Response Example
N/A
```

--------------------------------

### Cache Resources with @st.cache_resource in Python

Source: https://docs.streamlit.io/develop/api-reference/caching-and-state

The @st.cache_resource decorator caches functions that return global resources like database connections or machine learning models. This ensures that these resources are initialized only once, improving startup time and reducing memory usage.

```python
@st.cache_resource
def init_model():
  # Return a global resource here
  return pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
  )
```

--------------------------------

### Streamlit Authenticated Menu Logic (Python)

Source: https://docs.streamlit.io/develop/tutorials/multipage/st

Illustrates the specific logic within the `authenticated_menu` function for Streamlit applications. It shows how to add sidebar links for different user roles, including conditional display and disabling of links based on the user's role stored in `st.session_state`.

```Python
def authenticated_menu():
    # Show a navigation menu for authenticated users
    st.sidebar.page_link("app.py", label="Switch accounts")
    st.sidebar.page_link("pages/user.py", label="Your profile")
    if st.session_state.role in ["admin", "super-admin"]:
        st.sidebar.page_link("pages/admin.py", label="Manage users")
        st.sidebar.page_link(
            "pages/super-admin.py",
            label="Manage admin access",
            disabled=st.session_state.role != "super-admin",
        )

```

--------------------------------

### Displaying Markdown with st.markdown

Source: https://docs.streamlit.io/develop/api-reference_slug=private-gsheet

Render strings formatted as Markdown using the `st.markdown` function.

```APIDOC
## POST /st.markdown

### Description
Display string formatted as Markdown.

### Method
POST

### Endpoint
/st.markdown

### Parameters
#### Request Body
- **body** (string) - Required - The Markdown string to display.

### Request Example
```json
{
  "body": "Hello **world**!"
}
```

### Response
#### Success Response (200)
- **status** (string) - Indicates successful rendering of the Markdown.

#### Response Example
```json
{
  "status": "success"
}
```
```

--------------------------------

### AppTest - TextArea Interaction

Source: https://docs.streamlit.io/develop/api-reference_slug=%28&slug=develop&slug=concepts&slug=configuration

Simulates inputting text into the `st.text_area` widget.

```APIDOC
## TextArea

### Description
A representation of `st.text_area`.

### Method
Python

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
at.text_area[0].input("Streamlit is awesome!").run()
```

### Response
#### Success Response (200)
N/A

#### Response Example
N/A
```

--------------------------------

### Alternative Login Button Implementation in Streamlit

Source: https://docs.streamlit.io/develop/tutorials/authentication/google

This Python code demonstrates an alternative way to implement the login button using an `if` statement instead of the `on_click` callback. It checks if the button press event has occurred and then calls `st.login()` to initiate the authentication process. This provides flexibility in handling button interactions.

```python
if st.button("Log in with Google"):
   st.login()
```

--------------------------------

### Display Media

Source: https://docs.streamlit.io/develop/quick-reference/cheat-sheet

Functions for displaying images, logos, PDFs, audio, and video content.

```APIDOC
## Display Media

### Description
Embed various media types directly into your Streamlit application.

### Method
Python

### Endpoint
N/A

### Parameters
None

### Request Example
```python
st.image("./header.png")
st.logo("logo.jpg")
st.pdf("my_document.pdf")
st.audio(data)
st.video(data)
st.video(data, subtitles="./subs.vtt")
```

### Response
N/A
```

--------------------------------

### NLU (Natural Language Understanding)

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=configuration&slug=data

Applies text mining and natural language understanding on a pandas DataFrame. Created by @JohnSnowLabs.

```APIDOC
## NLU (Natural Language Understanding)

### Description
Apply text mining on a dataframe. Created by @JohnSnowLabs.

### Method
N/A (This is a component usage example)

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
nlu.load("sentiment").predict("I love NLU! <3")
```

### Response
#### Success Response (200)
N/A (Component interaction)

#### Response Example
N/A
```

--------------------------------

### Streamlit Embed Options with Query Parameters

Source: https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app/embed-your-app

Demonstrates how to use the `?embed_options` query parameter in JavaScript to control various aspects of an embedded Streamlit app. These options affect the visibility of the toolbar, padding, footer, colored line, loading screen, scrolling, and theme.

```javascript
/?embed=true&embed_options=show_toolbar
```

```javascript
/?embed=true&embed_options=show_padding
```

```javascript
/?embed=true&embed_options=show_footer
```

```javascript
/?embed=true&embed_options=show_colored_line
```

```javascript
/?embed=true&embed_options=hide_loading_screen
```

```javascript
/?embed=true&embed_options=disable_scrolling
```

```javascript
/?embed=true&embed_options=light_theme
```

```javascript
/?embed=true&embed_options=dark_theme
```

--------------------------------

### Enable Static File Serving in Streamlit

Source: https://docs.streamlit.io/develop/tutorials/configuration-and-theming/static-fonts

TOML configuration snippet to enable Streamlit's static file server. This allows files in the 'static' directory to be accessible via the app's URL.

```toml
[server]
enableStaticServing = true
```

--------------------------------

### Save Feedback Value to Streamlit Session State

Source: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/chat-response-feedback

Saves the retrieved feedback value into Streamlit's session state using a unique key derived from the message index. This prepares the feedback to be displayed and managed by a feedback widget.

```python
st.session_state[f"feedback_{i}"] = feedback
```

--------------------------------

### App Logic and Configuration

Source: https://docs.streamlit.io/develop/api-reference_slug=advanced-features&slug=prerelease

Manage user authentication, navigation, and script execution flow within your Streamlit application.

```APIDOC
## App Logic and Configuration

### Description
This section covers core Streamlit functionalities for structuring your application, managing user sessions, and controlling script execution.

### Authentication and User Info

#### Log in a user
Initiates an authentication flow with a configured identity provider.
```python
st.login()
```

#### Log out a user
Removes the current user's identity information, logging them out.
```python
st.logout()
```

#### User Info
Provides access to information about the currently logged-in user.
```python
if st.user.is_logged_in:
  st.write(f"Welcome back, {st.user.name}!")
```

### Navigation and Pages

#### Navigation
Configures the available pages in a multipage Streamlit application.
```python
# Assuming 'log_out', 'settings', 'overview', 'usage', 'search' are defined page objects
# st.navigation({
#     "Your account" : [log_out, settings],
#     "Reports" : [overview, usage],
#     "Tools" : [search]
# })
```

#### Page Definition
Defines a page within a multipage application, specifying its script, title, and icon.
```python
# Example page definition
# home = st.Page(
#     "home.py",
#     title="Home",
#     icon=":material/home:"
# )
```

#### Page Link
Creates a navigable link to another page within the application.
```python
# st.page_link("app.py", label="Home", icon="🏠")
# st.page_link("pages/profile.py", label="My profile")
```

#### Switch Page
Programmatically navigates the user to a specified page.
```python
# st.switch_page("pages/my_page.py")
```

### Execution Flow

#### Modal Dialog
Inserts a modal dialog that can be rendered and rerun independently.
```python
# @st.dialog("Sign up")
# def email_form():
#     name = st.text_input("Name")
#     email = st.text_input("Email")
```

#### Forms
Groups input widgets together with a "Submit" button, allowing for batched submission.
```python
# with st.form(key='my_form'):
#     name = st.text_input("Name")
#     email = st.text_input("Email")
#     st.form_submit_button("Sign up")
```

#### Fragments
Defines a section of the app that can rerun independently of the rest of the script.
```python
# @st.fragment(run_every="10s")
# def fragment():
#     df = get_data()
#     st.line_chart(df)
```

#### Rerun Script
Immediately reruns the entire Streamlit script.
```python
st.rerun()
```

#### Stop Execution
Halts the execution of the Streamlit script immediately.
```python
st.stop()
```
```

--------------------------------

### Organize Content with Tabs in Streamlit

Source: https://docs.streamlit.io/develop/api-reference/layout

This snippet demonstrates how to create tabs in Streamlit to organize content. Each tab can contain its own set of Streamlit elements.

```python
tab1, tab2 = st.tabs(["Tab 1", "Tab2"])
tab1.write("this is tab 1")
tab2.write("this is tab 2")
```

--------------------------------

### Display Raw Dataframe with st.write in Python

Source: https://docs.streamlit.io/get-started/tutorials/create-an-app

This Python code snippet shows how to display a raw dataframe within a Streamlit application using the st.write function. Streamlit's st.write is versatile and can render various data types, including dataframes, as interactive tables. For more control over table rendering, specialized commands like st.dataframe are also available.

```python
st.subheader('Raw data')
st.write(data)

```

--------------------------------

### Generate Streamlit UI from Pydantic Models

Source: https://docs.streamlit.io/develop/api-reference/execution-flow

Utilizes the streamlit-pydantic library by @lukasmasuch to automatically generate Streamlit forms from Pydantic models or Dataclasses. This simplifies data input UI creation.

```python
import streamlit_pydantic as sp

# Assume ExampleModel is a defined Pydantic model
schema = ExampleModel # Replace with your actual Pydantic model
sp.pydantic_form(key="my_form",
  model=schema)
```

--------------------------------

### Import and Cache Stock Data using Streamlit

Source: https://docs.streamlit.io/develop/tutorials/elements/annotate-an-altair-chart

Imports stock data from vega_datasets and caches it using Streamlit's `@st.cache_data` decorator to ensure data is downloaded only once. Filters data to include dates after '2004-01-01'.

```Python
import streamlit as st
import altair as alt
from vega_datasets import data
import pandas as pd

@st.cache_data
def get_data():
    source = data.stocks()
    source = source[source.date.gt("2004-01-01")]
    return source

stock_data = get_data()
```

--------------------------------

### TextInput Testing

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

Simulates interactions with the `st.text_input` element.

```APIDOC
## TextInput Testing

### Description
A representation of `st.text_input`.

### Method
Python

### Endpoint
N/A

### Parameters
#### Path Parameters
N/A

#### Query Parameters
N/A

#### Request Body
N/A

### Request Example
```python
at.text_input[0].input("Streamlit").run()
```

### Response
#### Success Response (200)
N/A

#### Response Example
N/A
```

--------------------------------

### TokensProxy - Accessing User Tokens

Source: https://docs.streamlit.io/develop/api-reference/user/st

The TokensProxy object provides read-only access to tokens exposed via the `expose_tokens` setting in your authentication configuration. Tokens can be accessed using dictionary or attribute notation.

```APIDOC
## GET /user/tokens

### Description
Access exposed identity and/or access tokens for the logged-in user.

### Method
GET

### Endpoint
/user/tokens

### Parameters
#### Query Parameters
None

#### Request Body
None

### Request Example
```python
import streamlit as st

if st.user.is_logged_in:
    id_token = st.user.tokens.get("id") # or st.user.tokens["id"]
    access_token = st.user.tokens.get("access") # or st.user.tokens["access"]
    # Use tokens for server-side API calls
```

### Response
#### Success Response (200)
- **id** (str) - The identity token. Available if 'id' is in `expose_tokens`.
- **access** (str) - The access token. Available if 'access' is in `expose_tokens`.

#### Response Example
```json
{
  "id": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "access": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

### Configuration
To expose tokens, configure `expose_tokens` in your `.streamlit/secrets.toml` file under the `[auth]` section.

**Example 1: Expose ID token**
```toml
[auth]
expose_tokens = "id"
```

**Example 2: Expose ID and access tokens**
```toml
[auth]
expose_tokens = ["id", "access"]
```
```

--------------------------------

### Integrate Lottie Animation

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=configuration&slug=chat

This code snippet demonstrates how to integrate Lottie animations into a Streamlit application using the `streamlit-lottie` library. It loads a Lottie animation from a URL and displays it using `st_lottie`.

```Python
from streamlit_lottie import st_lottie
lottie_hello = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_V9t630.json")
st_lottie(lottie_hello, key="hello")
```

--------------------------------

### User Authentication in Streamlit

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

This Python code snippet uses the `streamlit_authenticator` library to implement user authentication. It initializes the authenticator with configuration for credentials, cookies, and preauthorized users.

```python
import streamlit_authenticator as stauth

authenticator = stauth.Authenticate( config['credentials'], config['cookie']['name'],
config['cookie']['key'], config['cookie']['expiry_days'], config['preauthorized'])
```

--------------------------------

### Implement Popover Containers in Streamlit

Source: https://docs.streamlit.io/develop/api-reference/layout

This snippet shows how to use `st.popover` to create a popover container that can be opened and closed. It's useful for displaying settings or additional information without cluttering the main layout.

```python
with st.popover("Settings"):
  st.checkbox("Show completed")
```

--------------------------------

### Run Streamlit as Python Module

Source: https://docs.streamlit.io/develop/concepts/architecture/run-your-app

An alternative method to run Streamlit apps by executing it as a Python module. This is often used for IDE configurations, like in PyCharm.

```bash
python -m streamlit run your_script.py
```

--------------------------------

### Generate Line Chart with Plost

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=configuration&slug=data

Creates a line chart using the Plost library, a simple plotting tool for Streamlit. It visualizes stock data with time, value, and name dimensions.

```python
import plost
import streamlit as st

# Assuming my_dataframe is a pre-defined Pandas DataFrame
# plost.line_chart(my_dataframe, x='time', y='stock_value', color='stock_name')
```

--------------------------------

### Streamlit Extras for Enhanced UI

Source: https://docs.streamlit.io/develop/api-reference_slug=%28&slug=develop&slug=concepts&slug=configuration

Provides a collection of useful extras for Streamlit applications, including styling for metric cards. Requires the streamlit-extras library.

```Python
from streamlit_extras.metric_cards import style_metric_cards
import streamlit as st

# Assuming 'col3' is a defined streamlit column
col3 = st.columns(1)[0]
col3.metric(label="No Change", value=5000, delta=0)

style_metric_cards()
```

--------------------------------

### Highlight Max Values in DataFrame with st.dataframe() and Pandas Styler

Source: https://docs.streamlit.io/get-started/fundamentals/main-concepts

Illustrates using st.dataframe() with a Pandas Styler object to highlight the maximum value in each column of a DataFrame. This requires both Streamlit and Pandas libraries.

```python
import streamlit as st
import numpy as np
import pandas as pd

dataframe = pd.DataFrame(
    np.random.randn(10, 20),
    columns=('col %d' % i for i in range(20)))

st.dataframe(dataframe.style.highlight_max(axis=0))
```

--------------------------------

### App Testing with AppTest

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=configuration&slug=status

Introduces `st.testing.v1.AppTest` for simulating and testing Streamlit applications.

```APIDOC
## Developer Tools: App Testing

### Description
`st.testing.v1.AppTest` simulates a running Streamlit app, enabling automated testing.

### Initialization
`AppTest` can be initialized from a file, a string, or a function.

#### `AppTest.from_file(filepath)`
Initializes a simulated app from a Python file.
```python
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("streamlit_app.py")
at.run()
```

#### `AppTest.from_string(app_script_as_string)`
Initializes a simulated app from a string containing the app script.
```python
from streamlit.testing.v1 import AppTest
at = AppTest.from_string(app_script_as_string)
at.run()
```

#### `AppTest.from_function(app_script_as_callable)`
Initializes a simulated app from a callable function.
```python
from streamlit.testing.v1 import AppTest
at = AppTest.from_function(app_script_as_callable)
at.run()
```

### Interacting with the App
`AppTest` provides methods to interact with app elements and assert their states.

#### Setting Secrets
```python
at.secrets["WORD"] = "Foobar"
```

#### Running the App
```python
at.run()
```

#### Asserting No Exceptions
```python
assert not at.exception
```

### Element Interaction Examples

#### Block (`st.chat_message`, `st.columns`, etc.)
```python
# at.sidebar returns a Block
at.sidebar.button[0].click().run()
assert not at.exception
```

#### Element (Base class for `st.title`, `st.header`, etc.)
```python
# at.title returns a sequence of Title
# Title inherits from Element
assert at.title[0].value == "My awesome app"
```

#### Button (`st.button`, `st.form_submit_button`)
```python
at.button[0].click().run()
```

#### ChatInput (`st.chat_input`)
```python
at.chat_input[0].set_value("What is Streamlit?").run()
```

#### Checkbox (`st.checkbox`)
```python
at.checkbox[0].check().run()
```

#### ColorPicker (`st.color_picker`)
```python
at.color_picker[0].pick("#FF4B4B").run()
```

#### DateInput (`st.date_input`)
```python
import datetime
release_date = datetime.date(2023, 10, 26)
at.date_input[0].set_value(release_date).run()
```

#### Multiselect (`st.multiselect`)
```python
at.multiselect[0].select("New York").run()
```

#### NumberInput (`st.number_input`)
```python
at.number_input[0].increment().run()
```

#### Radio (`st.radio`)
```python
at.radio[0].set_value("New York").run()
```

#### SelectSlider (`st.select_slider`)
```python
at.select_slider[0].set_range("A","C").run()
```

#### Selectbox (`st.selectbox`)
```python
at.selectbox[0].select("New York").run()
```

#### Slider (`st.slider`)
```python
at.slider[0].set_range(2,5).run()
```

#### TextArea (`st.text_area`)
```python
at.text_area[0].input("Streamlit is awesome!").run()
```

#### TextInput (`st.text_input`)
```python
at.text_input[0].input("Streamlit").run()
```

#### TimeInput (`st.time_input`)
```python
at.time_input[0].increment().run()
```

#### Toggle (`st.toggle`)
```python
at.toggle[0].set_value("True").run()
```

### Response
#### Success Response (200)
- **status** (string) - Indicates the test run was successful.
```

--------------------------------

### TimeInput Testing

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

Simulates interactions with the `st.time_input` element.

```APIDOC
## TimeInput Testing

### Description
A representation of `st.time_input`.

### Method
Python

### Endpoint
N/A

### Parameters
#### Path Parameters
N/A

#### Query Parameters
N/A

#### Request Body
N/A

### Request Example
```python
at.time_input[0].increment().run()
```

### Response
#### Success Response (200)
N/A

#### Response Example
N/A
```

--------------------------------

### Interactive Widgets

Source: https://docs.streamlit.io/develop/quick-reference/cheat-sheet

A comprehensive list of interactive widgets available in Streamlit for user input and interaction.

```APIDOC
## Display Interactive Widgets

### Description
Streamlit offers a wide array of interactive widgets for collecting user input, ranging from simple buttons to complex data editors and file uploaders.

### Method
Python

### Endpoint
N/A

### Parameters
None

### Request Example
```python
st.button("Click me")
st.download_button("Download file", data)
st.link_button("Go to gallery", url)
st.page_link("app.py", label="Home")
st.data_editor("Edit data", data)
st.checkbox("I agree")
st.feedback("thumbs")
st.pills("Tags", ["Sports", "Politics"])
st.radio("Pick one", ["cats", "dogs"])
st.segmented_control("Filter", ["Open", "Closed"])
st.toggle("Enable")
st.selectbox("Pick one", ["cats", "dogs"])
st.multiselect("Buy", ["milk", "apples", "potatoes"])
st.slider("Pick a number", 0, 100)
st.select_slider("Pick a size", ["S", "M", "L"])
st.text_input("First name")
st.number_input("Pick a number", 0, 10)
st.text_area("Text to translate")
st.date_input("Your birthday")
st.datetime_input("Event date and time")
st.time_input("Meeting time")
st.file_uploader("Upload a CSV")
st.audio_input("Record a voice message")
st.camera_input("Take a picture")
st.color_picker("Pick a color")
```

### Response
N/A
```

--------------------------------

### Configuration API

Source: https://docs.streamlit.io/develop/api-reference

APIs for configuring Streamlit application settings.

```APIDOC
## Configuration File

### Description
Configures the default settings for your app.

### Method
File Structure

### Endpoint
`your-project/.streamlit/config.toml`

## Get Config Option

### Description
Retrieve a single configuration option.

### Method
Python

### Endpoint
`st.get_option`

### Parameters
#### Path Parameters
- **option** (str) - Required - The configuration option to retrieve (e.g., "theme.primaryColor").

### Request Example
```python
st.get_option("theme.primaryColor")
```

## Set Config Option

### Description
Set a single configuration option. (This is very limited.)

### Method
Python

### Endpoint
`st.set_option`

### Parameters
#### Path Parameters
- **option** (str) - Required - The configuration option to set.
- **value** (any) - Required - The value to set for the option.

### Request Example
```python
st.set_option("deprecation.showPyplotGlobalUse", False)
```

## Set Page Config

### Description
Configures the default settings of the page, including title and favicon.

### Method
Python

### Endpoint
`st.set_page_config`

### Parameters
#### Path Parameters
- **page_title** (str) - Optional - The title of the browser tab.
- **page_icon** (str) - Optional - The icon for the browser tab (e.g., emoji or path to an image).

### Request Example
```python
st.set_page_config(
  page_title="My app",
  page_icon=":shark:",
)
```
```

--------------------------------

### Streamlit App to Query MySQL Database

Source: https://docs.streamlit.io/develop/tutorials/databases/mysql

Python code for a Streamlit app that connects to a MySQL database using `st.connection`, queries data from a table, and displays the results. It demonstrates basic data retrieval and display with caching.

```python
# streamlit_app.py

import streamlit as st

# Initialize connection.
conn = st.connection('mysql', type='sql')

# Perform query.
df = conn.query('SELECT * from mytable;', ttl=600)

# Print results.
for row in df.itertuples():
    st.write(f"{row.name} has a :{row.pet}:")
```

--------------------------------

### Configuration File Structure

Source: https://docs.streamlit.io/develop/api-reference_slug=%28&slug=develop&slug=concepts&slug=configuration

Defines the structure for configuring default settings of your Streamlit app using a TOML file.

```APIDOC
## Configuration File

### Description
Configures the default settings for your app.

### Method
File Structure

### Endpoint
N/A

### Parameters
None

### Request Example
```
your-project/
├── .streamlit/
│   └── config.toml
└── your_app.py
```

### Response
N/A
```

--------------------------------

### Display a feedback widget in Python

Source: https://docs.streamlit.io/develop/api-reference/widgets

Renders a feedback mechanism, such as star ratings or sentiment buttons. Allows users to provide quick feedback on content or features. The type of feedback is specified by a string argument.

```python
st.feedback("stars")
```

--------------------------------

### Configuring Streamlit Server Base Path

Source: https://docs.streamlit.io/develop/quick-reference/release-notes/2019

Allows setting Streamlit's URL to a custom path using the `server.baseUrlPath` configuration option. This is useful for deploying Streamlit apps under a specific subdirectory.

```bash
streamlit config set server.baseUrlPath /customPath
```

--------------------------------

### Simulate Radio Button Selection (Python)

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

Simulates selecting an option from a radio button group using `AppTest`. This allows testing of single-choice selections.

```python
at.radio[0].set_value("New York").run()
```

--------------------------------

### Create Expandable Content Sections in Streamlit

Source: https://docs.streamlit.io/develop/api-reference/layout

This code demonstrates how to use `st.expander` to create collapsible sections in Streamlit. Content within the `with` block is hidden until the user expands the section.

```python
with st.expander("Open to see more"):
  st.write("This is more content")
```

--------------------------------

### Streamlit App Layout: Title and Containers

Source: https://docs.streamlit.io/develop/tutorials/execution-flow/create-a-multiple-container-fragment

Sets up the basic layout of a Streamlit app by adding a title and creating a grid of containers. This is foundational for organizing content within the app.

```python
st.title("Cats!")

row1 = st.columns(3)
row2 = st.columns(3)

grid = [col.container(height=200) for col in row1 + row2]
```

--------------------------------

### Streamlit Text Display Functions

Source: https://docs.streamlit.io/develop/quick-reference/cheat-sheet

Demonstrates various Streamlit functions for displaying text content, including standard text, Markdown, LaTeX, titles, headers, subheaders, code blocks, badges, and raw HTML.

```python
st.write("Most objects") # df, err, func, keras!
st.write(["st", "is <", 3])
st.write_stream(my_generator)
st.write_stream(my_llm_stream)

st.text("Fixed width text")
st.markdown("_Markdown_")
st.latex(r""" e^{i\pi} + 1 = 0 """)
st.title("My title")
st.header("My header")
st.subheader("My sub")
st.code("for i in range(8): foo()")
st.badge("New")
st.html("<p>Hi!</p>")
```

--------------------------------

### NumberInput Testing

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

Simulates interactions with the `st.number_input` element.

```APIDOC
## NumberInput Testing

### Description
A representation of `st.number_input`.

### Method
Python

### Endpoint
N/A

### Parameters
#### Path Parameters
N/A

#### Query Parameters
N/A

#### Request Body
N/A

### Request Example
```python
at.number_input[0].increment().run()
```

### Response
#### Success Response (200)
N/A

#### Response Example
N/A
```

--------------------------------

### Display a button widget in Python

Source: https://docs.streamlit.io/develop/api-reference/widgets

Renders a clickable button on the Streamlit app. Returns True when the button is clicked, otherwise False. No external dependencies are required.

```python
clicked = st.button("Click me")
```

--------------------------------

### Expose Streamlit Port in Docker

Source: https://docs.streamlit.io/deploy/tutorials/docker

Informs Docker that the container will listen on port 8501 at runtime. This is the default port used by Streamlit applications.

```dockerfile
EXPOSE 8501
```

--------------------------------

### Display a radio button widget in Python

Source: https://docs.streamlit.io/develop/api-reference/widgets

Presents a list of options where only one can be selected at a time. Returns the value of the selected radio button. Ideal for mutually exclusive choices.

```python
choice = st.radio("Pick one", ["cats", "dogs"])
```

--------------------------------

### Live Camera Input Component for Streamlit

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

Offers an alternative to `st.camera_input` that provides live webcam image feed. This component captures and returns webcam images in real-time.

```python
from camera_input_live import camera_input_live
import streamlit as st

image = camera_input_live()
# 'value' is likely a variable holding the image data from camera_input_live()
# st.image(value) # Uncomment and replace 'value' if needed
```

--------------------------------

### Set Streamlit App Title

Source: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/llm-quickstart

Sets the main title for the Streamlit application using the `st.title` function. This title will be displayed prominently at the top of the web page.

```python
st.title("🦜🔗 Quickstart App")
```

--------------------------------

### Display 'Accept' button in Streamlit

Source: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/validate-and-edit-chat-responses

This Python code displays a standard button labeled 'Accept' within the second column. When clicked, this button signifies the user's approval of the pending response. It triggers a state update to move the response to chat history and resets pending and validation states before rerunning the app.

```python
if cols[1].button("Accept"):
```

--------------------------------

### Button Interaction

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=configuration&slug=layouts

Simulates user interaction with buttons like `st.button` and `st.form_submit_button`.

```APIDOC
## Button

### Description
A representation of `st.button` and `st.form_submit_button`.

### Method
Python (AppTest interaction)

### Endpoint
N/A

### Parameters
None

### Request Example
```python
at.button[0].click().run()
```

### Response
N/A

### Response Example
N/A
```

--------------------------------

### Spacy-Streamlit for NLP Visualization

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

Provides spaCy building blocks and visualizers for natural language processing tasks in Streamlit apps. Requires 'spacy-streamlit' and spaCy models.

```python
import streamlit as st
import spacy_streamlit

models = ["en_core_web_sm", "en_core_web_md"]
spacy_streamlit.visualize(models, "Sundar Pichai is the CEO of Google.")
```

--------------------------------

### Shared Session State Across Streamlit Pages

Source: https://docs.streamlit.io/develop/concepts/multipage-apps/pages-directory

Illustrates how `st.session_state` is shared across all pages in a Streamlit multipage app. Data stored in `st.session_state` on one page is accessible on other pages, enabling persistent state management throughout the user's session.

```python
# page1.py
import streamlit as st
if "shared" not in st.session_state:
   st.session_state["shared"] = True

# page2.py
import streamlit as st
st.write(st.session_state["shared"]) # If page1 already executed, this writes True
```

--------------------------------

### Displaying Vega-Lite Charts in Streamlit

Source: https://docs.streamlit.io/develop/api-reference_slug=%28&slug=develop&slug=concepts&slug=configuration

Embeds Vega-Lite visualizations within Streamlit apps. Requires a Vega-Lite chart specification.

```Python
import streamlit as st

# Assuming my_vega_lite_chart is a Vega-Lite chart specification (often a dictionary)
# Example:
# my_vega_lite_chart = {
#   "data": {"values": [{"x": 1, "y": 2}, {"x": 2, "y": 4}]},
#   "mark": "line",
#   "encoding": {"x": "x", "y": "y"}
# }

st.vega_lite_chart(my_vega_lite_chart)
```

--------------------------------

### Create Basic Line Chart with Altair

Source: https://docs.streamlit.io/develop/tutorials/elements/annotate-an-altair-chart

Generates a basic line chart using Altair to visualize stock prices over time. It encodes the x-axis as 'date', the y-axis as 'price', and uses 'symbol' to differentiate multiple stock lines. Includes a title for the chart.

```Python
lines = (
    alt.Chart(stock_data, title="Evolution of stock prices")
    .mark_line()
    .encode(
        x="date",
        y="price",
        color="symbol",
    )
)
```

--------------------------------

### Display a datetime input widget in Python

Source: https://docs.streamlit.io/develop/api-reference/widgets

Allows users to select both a date and a time. Returns a `datetime.datetime` object. Ideal for scheduling events or time-sensitive inputs.

```python
datetime = st.datetime_input("Schedule your event")
```

--------------------------------

### Set Page Configuration

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=configuration&slug=chat

Configures the default settings of the Streamlit page, including title and favicon.

```APIDOC
## POST /api/set_page_config

### Description
Configures the default settings of the page, such as title and icon.

### Method
POST

### Endpoint
`/api/set_page_config`

### Parameters
#### Request Body
- **page_title** (string) - Optional - The title of the browser tab.
- **page_icon** (string) - Optional - The favicon for the page (e.g., an emoji or a path to an image).

### Request Example
```python
st.set_page_config(
  page_title="My app",
  page_icon=":shark:",
)
```

### Response
#### Success Response (200)
- **status** (string) - Indicates the page configuration was updated.
```

--------------------------------

### Create Tabs and Call Methods Directly in Streamlit

Source: https://docs.streamlit.io/develop/api-reference/layout/st

Shows an alternative method for using Streamlit's st.tabs where methods are called directly on the returned tab objects. This approach is useful for directly associating elements with specific tabs.

```python
import streamlit as st
from numpy.random import default_rng as rng

df = rng(0).standard_normal((10, 1))

tab1, tab2 = st.tabs(["📈 Chart", "🗃 Data"])

tab1.subheader("A tab with a chart")
tab1.line_chart(df)

tab2.subheader("A tab with the data")
tab2.write(df)
```

--------------------------------

### Test Streamlit Input Widgets with AppTest

Source: https://docs.streamlit.io/develop/concepts/app-testing/cheat-sheet

This code demonstrates testing various input widgets in Streamlit, such as buttons, checkboxes, color pickers, date inputs, multiselects, number inputs, radio buttons, select boxes, sliders, text areas, text inputs, and toggles. It shows how to interact with these widgets (e.g., clicking, checking, setting values) and assert their states.

```python
from streamlit.testing.v1 import AppTest
import datetime

at = AppTest.from_file("cheatsheet_app.py")

# button
assert at.button[0].value == False
at.button[0].click().run()
assert at.button[0].value == True

# checkbox
assert at.checkbox[0].value == False
at.checkbox[0].check().run() # uncheck() is also supported
assert at.checkbox[0].value == True

# color_picker
assert at.color_picker[0].value == "#FFFFFF"
at.color_picker[0].pick("#000000").run()

# date_input
assert at.date_input[0].value == datetime.date(2019, 7, 6)
at.date_input[0].set_value(datetime.date(2022, 12, 21)).run()

# form_submit_button - shows up just like a button
assert at.button[0].value == False
at.button[0].click().run()
assert at.button[0].value == True

# multiselect
assert at.multiselect[0].value == ["foo", "bar"]
at.multiselect[0].select("baz").unselect("foo").run()

# number_input
assert at.number_input[0].value == 5
at.number_input[0].increment().run()

# radio
assert at.radio[0].value == "Bar"
assert at.radio[0].index == 3
at.radio[0].set_value("Foo").run()

# selectbox
assert at.selectbox[0].value == "Bar"
assert at.selectbox[0].index == 3
at.selectbox[0].set_value("Foo").run()

# select_slider
assert at.select_slider[0].value == "Feb"
at.select_slider[0].set_value("Mar").run()
at.select_slider[0].set_range("Apr", "Jun").run()

# slider
assert at.slider[0].value == 2
at.slider[0].set_value(3).run()
at.slider[0].set_range(4, 6).run()

# text_area
assert at.text_area[0].value == "Hello, world!"
at.text_area[0].set_value("Hello, yourself!").run()

# text_input
assert at.text_input[0].value == "Hello, world!"
at.text_input[0].set_value("Hello, yourself!").run()

# time_input
assert at.time_input[0].value == datetime.time(8, 45)
at.time_input[0].set_value(datetime.time(12, 30))

# toggle
assert at.toggle[0].value == False
assert at.toggle[0].label == "Debug mode"
at.toggle[0].set_value(True).run()
assert at.toggle[0].value == True
```

--------------------------------

### Display and execute code with st.echo

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

st.echo is a context manager that displays the code within its block in the app and then executes it. This is particularly useful for creating tutorials or demonstrating code execution.

```Python
with st.echo():
  st.write('This code will be printed')
```

--------------------------------

### Streamlit Extras for Enhanced UI

Source: https://docs.streamlit.io/develop/api-reference

Provides a collection of useful UI elements and enhancements for Streamlit applications, including styled metric cards. Requires the streamlit_extras library.

```python
import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards

# Assuming 'col3' is a Streamlit column object
# col3.metric(label="No Change", value=5000, delta=0)

style_metric_cards()
```

--------------------------------

### Add Streamlit Feedback Widget to Chat Message

Source: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/chat-response-feedback

Adds a feedback widget (e.g., 'thumbs') to a chat message container. It uses a session state key to manage the feedback value and disables the widget if feedback has already been provided.

```python
st.feedback(
    "thumbs",
    key=f"feedback_{i}",
    disabled=feedback is not None,
)
```

--------------------------------

### Detect Theme Mode (Light/Dark) with st.context.theme

Source: https://docs.streamlit.io/develop/quick-reference/release-notes/2025

Allows developers to detect the current theme mode (light or dark) of the Streamlit app at runtime. This is useful for dynamically adjusting UI elements or content based on the user's preference. Access the theme context using `st.context.theme`.

```python
import streamlit as st

if st.context.theme.mode == "dark":
    st.write("Dark mode is enabled")
else:
    st.write("Light mode is enabled")
```

--------------------------------

### Display a single-line text input widget in Python

Source: https://docs.streamlit.io/develop/api-reference/widgets

Creates a text field for single-line text input. Returns the entered string. Commonly used for names, titles, or short text entries.

```python
name = st.text_input("First name")
```

--------------------------------

### Streamlit st.time_input Empty Initial Value

Source: https://docs.streamlit.io/develop/api-reference/widgets/st

Shows how to initialize an st.time_input widget with no default value. By setting the 'value' parameter to None, the widget will initially be empty and return None until the user selects a time.

```python
import datetime
import streamlit as st

t = st.time_input("Set an alarm for", value=None)
st.write("Alarm is set for", t)
```

--------------------------------

### Parse and highlight response errors in Streamlit

Source: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/validate-and-edit-chat-responses

This Python code snippet demonstrates parsing response sentences and identifying validation errors using helper functions `validate` and `add_highlights`. It takes the pending response from `st.session_state.pending` and returns highlighted sentences, which are then prepared for display. This is a crucial step in the validation stage.

```python
response_sentences, validation_list = validate(st.session_state.pending)
highlighted_sentences = add_highlights(response_sentences, validation_list)
```

--------------------------------

### Admin Page Role Check in Streamlit

Source: https://docs.streamlit.io/develop/tutorials/multipage/st

This Streamlit page restricts access to users with 'admin' or 'super-admin' roles. It uses `menu_with_redirect()` for authentication and then checks `st.session_state.role`. If the role is not permitted, a warning is displayed, and `st.stop()` halts further execution of the script for that user.

```python
import streamlit as st
from menu import menu_with_redirect

# Redirect to app.py if not logged in, otherwise show the navigation menu
menu_with_redirect()

# Verify the user's role
if st.session_state.role not in ["admin", "super-admin"]:
    st.warning("You do not have permission to view this page.")
    st.stop()

st.title("This page is available to all admins")
st.markdown(f"You are currently logged with the role of {st.session_state.role}.")
```

--------------------------------

### Display Vega-Lite Charts in Streamlit

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=configuration&slug=widgets

Integrates charts created with the Vega-Lite specification into Streamlit apps. It requires a Vega-Lite chart specification.

```python
import streamlit as st

# Assuming my_vega_lite_chart is a Vega-Lite chart specification (e.g., a dictionary)
# Example:
# my_vega_lite_chart = {
#   "data": {"values": [{"x": 1, "y": 4}, {"x": 2, "y": 5}, {"x": 3, "y": 6}]},
#   "mark": "line",
#   "encoding": {"x": "x", "y": "y"}
# }
st.vega_lite_chart(my_vega_lite_chart)
```

--------------------------------

### Toggle Testing

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

Simulates interactions with the `st.toggle` element.

```APIDOC
## Toggle Testing

### Description
A representation of `st.toggle`.

### Method
Python

### Endpoint
N/A

### Parameters
#### Path Parameters
N/A

#### Query Parameters
N/A

#### Request Body
N/A

### Request Example
```python
at.toggle[0].set_value("True").run()
```

### Response
#### Success Response (200)
N/A

#### Response Example
N/A
```

--------------------------------

### Define Callback to Save Feedback in Streamlit

Source: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/chat-response-feedback

Defines a callback function `save_feedback` that takes a message index. This function updates the chat history in session state with the feedback value from the corresponding feedback widget.

```python
def save_feedback(index):
    st.session_state.history[index]["feedback"] = st.session_state[f"feedback_{index}"]
```

--------------------------------

### Streamlit Extras - Mention

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=configuration&slug=data

A library with useful Streamlit extras, including the 'mention' component for creating links with icons.

```APIDOC
## Streamlit Extras - Mention

### Description
A library with useful Streamlit extras. Created by @arnaudmiribel.

### Method
N/A (This is a component usage example)

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
mention(label="An awesome Streamlit App", icon="streamlit",  url="https://extras.streamlit.app")
```

### Response
#### Success Response (200)
N/A (Component interaction)

#### Response Example
N/A
```

--------------------------------

### Magic Commands

Source: https://docs.streamlit.io/develop/api-reference/write-magic

Streamlit's magic commands allow you to automatically display variables or literals by placing them on their own line, using st.write behind the scenes.

```APIDOC
## Magic Commands

### Description
Streamlit automatically writes variables or literal values to your app when they appear on their own line. This is equivalent to using `st.write()` for that specific element.

### Method
Implicit (placing variables/literals on separate lines)

### Endpoint
N/A (Client-side feature)

### Parameters
N/A

### Request Example
```python
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Displaying text using magic
"Hello **world**!"

# Displaying a DataFrame using magic
df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
df

# Displaying a Matplotlib figure using magic
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])
fig
```

### Response
#### Success Response (200)
N/A (Streamlit renders content directly)

#### Response Example
N/A
```

--------------------------------

### Display Progress Bar with stqdm

Source: https://docs.streamlit.io/develop/api-reference/status

Provides a simple way to add a progress bar to Streamlit apps using the `stqdm` library. It integrates seamlessly with iterable loops to show progress.

```python
from stqdm import stqdm

for _ in stqdm(range(50)):
    sleep(0.5)
```

--------------------------------

### Streamlit App Logic and Configuration

Source: https://docs.streamlit.io/develop/api-reference_slug=publish

Functions for managing authentication, navigation, and execution flow in Streamlit applications.

```APIDOC
## App Logic and Configuration

This section covers Streamlit functions for authentication, navigation, and controlling script execution.

### Authentication and User Info

#### `st.login`

**Description**: Starts an authentication flow with an identity provider.

**Method**: Python Function Call

**Endpoint**: N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
st.login()
```

### Response
#### Success Response (N/A)
N/A

#### Response Example
N/A

---

#### `st.logout`

**Description**: Removes a user's identity information.

**Method**: Python Function Call

**Endpoint**: N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
st.logout()
```

### Response
#### Success Response (N/A)
N/A

#### Response Example
N/A

---

#### `st.user`

**Description**: Returns information about a logged-in user.

**Method**: Python Attribute

**Endpoint**: N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
if st.user.is_logged_in:
  st.write(f"Welcome back, {st.user.name}!")
```

### Response
#### Success Response (N/A)
N/A

#### Response Example
N/A

---

### Navigation and Pages

#### `st.navigation`

**Description**: Configures the available pages in a multipage app.

**Method**: Python Function Call

**Endpoint**: N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
st.navigation({
    "Your account" : [log_out, settings],
    "Reports" : [overview, usage],
    "Tools" : [search]
})
```

### Response
#### Success Response (N/A)
N/A

#### Response Example
N/A

---

#### `st.Page`

**Description**: Defines a page in a multipage app.

**Method**: Python Class Instantiation

**Endpoint**: N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
home = st.Page(
    "home.py",
    title="Home",
    icon=":material/home:"
)
```

### Response
#### Success Response (N/A)
N/A

#### Response Example
N/A

---

#### `st.page_link`

**Description**: Displays a link to another page in a multipage app.

**Method**: Python Function Call

**Endpoint**: N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
st.page_link("app.py", label="Home", icon="🏠")
st.page_link("pages/profile.py", label="My profile")
```

### Response
#### Success Response (N/A)
N/A

#### Response Example
N/A

---

#### `st.switch_page`

**Description**: Programmatically navigates to a specified page.

**Method**: Python Function Call

**Endpoint**: N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
st.switch_page("pages/my_page.py")
```

### Response
#### Success Response (N/A)
N/A

#### Response Example
N/A

---

### Execution Flow

#### `st.dialog`

**Description**: Inserts a modal dialog that can rerun independently from the rest of the script.

**Method**: Python Decorator

**Endpoint**: N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
@st.dialog("Sign up")
def email_form():
    name = st.text_input("Name")
    email = st.text_input("Email")
```

### Response
#### Success Response (N/A)
N/A

#### Response Example
N/A

---

#### `st.form`

**Description**: Creates a form that batches elements together with a “Submit" button.

**Method**: Python Context Manager

**Endpoint**: N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
with st.form(key='my_form'):
    name = st.text_input("Name")
    email = st.text_input("Email")
    st.form_submit_button("Sign up")
```

### Response
#### Success Response (N/A)
N/A

#### Response Example
N/A

---

#### `st.fragment`

**Description**: Defines a fragment to rerun independently from the rest of the script.

**Method**: Python Decorator

**Endpoint**: N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
@st.fragment(run_every="10s")
def fragment():
    df = get_data()
    st.line_chart(df)
```

### Response
#### Success Response (N/A)
N/A

#### Response Example
N/A

---

#### `st.rerun`

**Description**: Reruns the script immediately.

**Method**: Python Function Call

**Endpoint**: N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
st.rerun()
```

### Response
#### Success Response (N/A)
N/A

#### Response Example
N/A

---

#### `st.stop`

**Description**: Stops execution immediately.

**Method**: Python Function Call

**Endpoint**: N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
st.stop()
```

### Response
#### Success Response (N/A)
N/A

#### Response Example
N/A
```

--------------------------------

### Interactive high-dimensional plotting with HiPlot

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=configuration&slug=text

Illustrates how to use HiPlot for high-dimensional interactive plotting. It takes a list of dictionaries, where each dictionary represents an experiment with parameters like 'dropout', 'lr', 'loss', and 'optimizer'. The `hip.Experiment.from_iterable` method creates an experiment object that can be displayed.

```python
import hiplot as hip
data = [{'dropout':0.1, 'lr': 0.001, 'loss': 10.0, 'optimizer': 'SGD'}, {'dropout':0.15, 'lr': 0.01, 'loss': 3.5, 'optimizer': 'Adam'}, {'dropout':0.3, 'lr': 0.1, 'loss': 4.5, 'optimizer': 'Adam'}]
hip.Experiment.from_iterable(data).display()
```

--------------------------------

### Simulate App Execution with AppTest (Python)

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

Utilizes `st.testing.v1.AppTest` to simulate a running Streamlit app for testing purposes. It allows interaction with app elements and assertion of outcomes.

```python
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("streamlit_app.py")
at.secrets["WORD"] = "Foobar"
at.run()
assert not at.exception

at.text_input("word").input("Bazbat").run()
assert at.warning[0].value == "Try again."
```

--------------------------------

### Interacting with App Elements

Source: https://docs.streamlit.io/develop/api-reference/app-testing

Provides methods to interact with various Streamlit elements within the simulated app, such as buttons, text inputs, and more.

```APIDOC
## Interacting with App Elements

### Description
Provides methods to interact with various Streamlit elements within the simulated app, such as buttons, text inputs, and more. Each element type has specific methods for manipulation and inspection.

### Element Interaction Examples

#### `Block` (e.g., `st.sidebar`, `st.columns`)
Represents container elements.

**Request Example**
```python
# at.sidebar returns a Block
at.sidebar.button[0].click().run()
assert not at.exception
```

#### `Element` (Base class for most elements)
Represents standard Streamlit elements like titles, headers, markdown, etc.

**Request Example**
```python
# at.title returns a sequence of Title
# Title inherits from Element
assert at.title[0].value == "My awesome app"
```

#### `Button` (`st.button`, `st.form_submit_button`)
Represents button elements.

**Request Example**
```python
at.button[0].click().run()
```

#### `ChatInput` (`st.chat_input`)
Represents chat input elements.

**Request Example**
```python
at.chat_input[0].set_value("What is Streamlit?").run()
```

#### `Checkbox` (`st.checkbox`)
Represents checkbox elements.

**Request Example**
```python
at.checkbox[0].check().run()
```

#### `ColorPicker` (`st.color_picker`)
Represents color picker elements.

**Request Example**
```python
at.color_picker[0].pick("#FF4B4B").run()
```

#### `DateInput` (`st.date_input`)
Represents date input elements.

**Request Example**
```python
import datetime
release_date = datetime.date(2023, 10, 26)
at.date_input[0].set_value(release_date).run()
```

#### `Multiselect` (`st.multiselect`)
Represents multiselect elements.

**Request Example**
```python
at.multiselect[0].select("New York").run()
```

#### `NumberInput` (`st.number_input`)
Represents number input elements.

**Request Example**
```python
at.number_input[0].increment().run()
```

#### `Radio` (`st.radio`)
Represents radio button elements.

**Request Example**
```python
at.radio[0].set_value("New York").run()
```

#### `SelectSlider` (`st.select_slider`)
Represents select slider elements.

**Request Example**
```python
at.select_slider[0].set_range("A","C").run()
```

#### `Selectbox` (`st.selectbox`)
Represents selectbox elements.

**Request Example**
```python
at.selectbox[0].select("New York").run()
```

#### `Slider` (`st.slider`)
Represents slider elements.

**Request Example**
```python
at.slider[0].set_range(2,5).run()
```

#### `TextArea` (`st.text_area`)
Represents text area elements.

**Request Example**
```python
at.text_area[0].input("Streamlit is awesome!").run()
```

#### `TextInput` (`st.text_input`)
Represents text input elements.

**Request Example**
```python
at.text_input[0].input("Streamlit").run()
```

#### `TimeInput` (`st.time_input`)
Represents time input elements.

**Request Example**
```python
at.time_input[0].increment().run()
```

#### `Toggle` (`st.toggle`)
Represents toggle elements.

**Request Example**
```python
at.toggle[0].set_value("True").run()
```
```

--------------------------------

### Use Snowpark Session with Streamlit

Source: https://docs.streamlit.io/develop/tutorials/databases/snowflake

This Python code snippet demonstrates how to establish a Snowpark session using Streamlit's connection object. It loads data from a Snowflake table into a Pandas DataFrame and iterates through it to display information. Caching is manually applied for performance.

```python
# streamlit_app.py

import streamlit as st

conn = st.connection("snowflake")

@st.cache_data
def load_table():
    session = conn.session()
    return session.table("mytable").to_pandas()

df = load_table()

for row in df.itertuples():
    st.write(f"{row.NAME} has a :{row.PET}:")
```

--------------------------------

### Declare Custom Component

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=configuration&slug=chat

This section shows how to create and register a custom component in Streamlit using Python.

```APIDOC
## POST /api/declare_component

### Description
Creates and registers a custom component.

### Method
POST

### Endpoint
`/api/declare_component`

### Parameters
#### Query Parameters
- **name** (string) - Required - The name of the custom component.
- **path** (string) - Required - The path to the component's frontend code.

### Request Example
```python
from st.components.v1 import declare_component
declare_component(
    "custom_slider",
    "/frontend",
)
```

### Response
#### Success Response (200)
- **message** (string) - Confirmation message that the component was declared.
```

--------------------------------

### Streamlit Connection Methods: close() and reset()

Source: https://docs.streamlit.io/develop/api-reference/connections/st.connections

Illustrates the usage of the close() and reset() methods within a Streamlit connection class. The close() method is for cleanup, while reset() reinitializes the connection upon next use. Note: reset() is deprecated in v1.54.0.

```python
class MyConnection(ExperimentalBaseConnection):
    # ... other methods ...

    def close(self):
        # Code to clean up the connection resources
        pass

    def reset(self):
        # Code to reset the connection for reinitialization
        pass

```

--------------------------------

### Test Streamlit Layouts and Containers with AppTest

Source: https://docs.streamlit.io/develop/concepts/app-testing/cheat-sheet

This code demonstrates how to test Streamlit's layout and container elements, specifically the sidebar, columns, and tabs. It shows how to access elements within these containers and assert their values.

```python
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("cheatsheet_app.py")

# sidebar
at.sidebar.text_input[0].set_value("Jane Doe")

# columns
at.columns[1].markdown[0].value == "Hello, world!"

# tabs
at.tabs[2].markdown[0].value == "Hello, yourself!"
```

--------------------------------

### Streamlit Extras Library

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=widget-semantics

Provides a collection of useful components and utilities for Streamlit applications, including styling for metric cards. Requires streamlit_extras.

```python
import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards

# Assuming col3 is a Streamlit column object
# col3.metric(label="No Change", value=5000, delta=0)

style_metric_cards()
```

--------------------------------

### Button Testing

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

Simulates interactions with `st.button` and `st.form_submit_button` elements.

```APIDOC
## Button Testing

### Description
A representation of `st.button` and `st.form_submit_button`.

### Method
Python

### Endpoint
N/A

### Parameters
#### Path Parameters
N/A

#### Query Parameters
N/A

#### Request Body
N/A

### Request Example
```python
at.button[0].click().run()
```

### Response
#### Success Response (200)
N/A

#### Response Example
N/A
```

--------------------------------

### Configure Allowed CORS Origins with server.enableCORS

Source: https://docs.streamlit.io/develop/quick-reference/release-notes/2025

Allows configuration of a list of allowed origins when Cross-Origin Resource Sharing (CORS) protection is enabled. This is crucial for security when your Streamlit app needs to be accessed from different domains. The configuration is typically done in `config.toml`.

```toml
[server]
enableCORS = true

[server.options]
allowedOrigins = ["https://example.com", "https://another.com"]
```

--------------------------------

### Set Streamlit Page Configuration

Source: https://docs.streamlit.io/develop/tutorials/execution-flow/trigger-a-full-script-rerun-from-a-fragment

Configures the Streamlit page to use a wide layout, maximizing the available screen space for displaying content. This is useful for dashboards or applications with multiple data visualizations.

```Python
import streamlit as st

st.set_page_config(layout="wide")
```

--------------------------------

### V1 Custom Components API

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=configuration&slug=charts

APIs for creating and registering custom Streamlit components, and for displaying HTML or loading external URLs in iframes.

```APIDOC
## Declare a component

### Description
Create and register a custom component.

### Method
Python

### Endpoint
`st.components.v1.declare_component`

### Parameters
#### Path Parameters
- **name** (str) - Required - The name of the component.
- **path** (str) - Required - The path to the component's frontend directory.

### Request Example
```python
from st.components.v1 import declare_component
declare_component(
    "custom_slider",
    "/frontend",
)
```

## HTML

### Description
Display an HTML string in an iframe.

### Method
Python

### Endpoint
`st.components.v1.html`

### Parameters
#### Path Parameters
- **content** (str) - Required - The HTML content to display.

### Request Example
```python
from st.components.v1 import html
html(
    "<p>Foo bar.</p>"
)
```

## iframe

### Description
Load a remote URL in an iframe.

### Method
Python

### Endpoint
`st.components.v1.iframe`

### Parameters
#### Path Parameters
- **url** (str) - Required - The URL to load in the iframe.

### Request Example
```python
from st.components.v1 import iframe
iframe(
    "docs.streamlit.io"
)
```
```

--------------------------------

### Execute SQL Query with st.cache_data

Source: https://docs.streamlit.io/develop/tutorials/databases/mssql

Executes a given SQL query against the established database connection and caches the results. The `@st.cache_data(ttl=600)` decorator ensures the query is rerun only when the query string changes or after a 10-minute TTL (time-to-live), preventing unnecessary database load.

```python
# Perform query.
# Uses st.cache_data to only rerun when the query changes or after 10 min.
@st.cache_data(ttl=600)
def run_query(query):
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()

rows = run_query("SELECT * from mytable;")

# Print results.
for row in rows:
    st.write(f"{row[0]} has a :{row[1]}:")
```

--------------------------------

### Enabling Linting with Ruff in Streamlit App Action

Source: https://docs.streamlit.io/develop/concepts/app-testing/automate-tests

Configures the 'streamlit-app-action' to perform linting on the Streamlit application code using Ruff. This helps identify and fix stylistic and programmatic errors, improving code quality.

```YAML
- uses: streamlit/streamlit-app-action@v0.0.3
  with:
    app-path: streamlit_app.py
    ruff: true

```

--------------------------------

### Streamlit App: Daily vs Monthly Sales Data Display

Source: https://docs.streamlit.io/develop/tutorials/execution-flow/trigger-a-full-script-rerun-from-a-fragment

This Python script creates a Streamlit application to visualize daily and monthly sales data. It uses `st.fragment` to isolate day-specific sales information, allowing for partial reruns when the user changes the selected day within the same month. A full script rerun is triggered using `st.rerun` when the selected day falls into a new month. The script includes data generation, daily sales display, and monthly sales aggregation.

```Python
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import string
import time


@st.cache_data
def get_data():
    """Generate random sales data for Widget A through Widget Z"""

    product_names = ["Widget " + letter for letter in string.ascii_uppercase]
    average_daily_sales = np.random.normal(1_000, 300, len(product_names))
    products = dict(zip(product_names, average_daily_sales))

    data = pd.DataFrame({})
    sales_dates = np.arange(date(2023, 1, 1), date(2024, 1, 1), timedelta(days=1))
    for product, sales in products.items():
        data[product] = np.random.normal(sales, 300, len(sales_dates)).round(2)
    data.index = sales_dates
    data.index = data.index.date
    return data


@st.fragment
def show_daily_sales(data):
    time.sleep(1)
    with st.container(height=100):
        selected_date = st.date_input(
            "Pick a day ",
            value=date(2023, 1, 1),
            min_value=date(2023, 1, 1),
            max_value=date(2023, 12, 31),
            key="selected_date",
        )

    if "previous_date" not in st.session_state:
        st.session_state.previous_date = selected_date
    previous_date = st.session_state.previous_date
    st.session_state.previous_date = selected_date
    is_new_month = selected_date.replace(day=1) != previous_date.replace(day=1)
    if is_new_month:
        st.rerun()

    with st.container(height=510):
        st.header(f"Best sellers, {selected_date:%m/%d/%y}")
        top_ten = data.loc[selected_date].sort_values(ascending=False)[0:10]
        cols = st.columns([1, 4])
        cols[0].dataframe(top_ten)
        cols[1].bar_chart(top_ten)

    with st.container(height=510):
        st.header(f"Worst sellers, {selected_date:%m/%d/%y}")
        bottom_ten = data.loc[selected_date].sort_values()[0:10]
        cols = st.columns([1, 4])
        cols[0].dataframe(bottom_ten)
        cols[1].bar_chart(bottom_ten)


def show_monthly_sales(data):
    time.sleep(1)
    selected_date = st.session_state.selected_date
    this_month = selected_date.replace(day=1)
    next_month = (selected_date.replace(day=28) + timedelta(days=4)).replace(day=1)

    st.container(height=100, border=False)
    with st.container(height=510):
        st.header(f"Daily sales for all products, {this_month:%B %Y}")
        monthly_sales = data[(data.index < next_month) & (data.index >= this_month)]
        st.write(monthly_sales)
    with st.container(height=510):
        st.header(f"Total sales for all products, {this_month:%B %Y}")
        st.bar_chart(monthly_sales.sum())


st.set_page_config(layout="wide")

st.title("Daily vs monthly sales, by product")
st.markdown("This app shows the 2023 daily sales for Widget A through Widget Z.")

data = get_data()
daily, monthly = st.columns(2)
with daily:
    show_daily_sales(data)
with monthly:
    show_monthly_sales(data)
```

--------------------------------

### Navigation and Pages

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=configuration&slug=media

APIs for configuring and managing navigation and pages in a multipage Streamlit application.

```APIDOC
## POST /st.navigation

### Description
Configures the available pages in a multipage app.

### Method
POST

### Endpoint
/st.navigation

### Parameters
#### Request Body
- **pages_config** (object) - Required - A configuration object defining the pages and their structure.

### Request Example
```json
{
  "pages_config": {
    "Your account": ["log_out", "settings"],
    "Reports": ["overview", "usage"],
    "Tools": ["search"]
  }
}
```

### Response
#### Success Response (200)
- **status** (string) - Indicates the operation was successful.

#### Response Example
```json
{
  "status": "success"
}
```
```

```APIDOC
## POST /st.Page

### Description
Defines a page in a multipage app.

### Method
POST

### Endpoint
/st.Page

### Parameters
#### Request Body
- **script_path** (string) - Required - The path to the page's script.
- **title** (string) - Optional - The title of the page.
- **icon** (string) - Optional - The icon for the page.

### Request Example
```json
{
  "script_path": "home.py",
  "title": "Home",
  "icon": ":material/home:"
}
```

### Response
#### Success Response (200)
- **page_object** (object) - The created page object.

#### Response Example
```json
{
  "page_object": "<Page object>"
}
```
```

```APIDOC
## POST /st.page_link

### Description
Displays a link to another page in a multipage app.

### Method
POST

### Endpoint
/st.page_link

### Parameters
#### Request Body
- **page** (string) - Required - The path to the target page.
- **label** (string) - Optional - The text to display for the link.
- **icon** (string) - Optional - The icon for the link.

### Request Example
```json
{
  "page": "app.py",
  "label": "Home",
  "icon": "🏠"
}
```

### Response
#### Success Response (200)
- **status** (string) - Indicates the operation was successful.

#### Response Example
```json
{
  "status": "success"
}
```
```

```APIDOC
## POST /st.switch_page

### Description
Programmatically navigates to a specified page.

### Method
POST

### Endpoint
/st.switch_page

### Parameters
#### Request Body
- **page_name** (string) - Required - The name or path of the page to switch to.

### Request Example
```json
{
  "page_name": "pages/my_page.py"
}
```

### Response
#### Success Response (200)
- **status** (string) - Indicates the operation was successful.

#### Response Example
```json
{
  "status": "success"
}
```
```

--------------------------------

### AppTest - ColorPicker Interaction

Source: https://docs.streamlit.io/develop/api-reference_slug=%28&slug=develop&slug=concepts&slug=configuration

Simulates selecting a color using the `st.color_picker` widget.

```APIDOC
## ColorPicker

### Description
A representation of `st.color_picker`.

### Method
Python

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
at.color_picker[0].pick("#FF4B4B").run()
```

### Response
#### Success Response (200)
N/A

#### Response Example
N/A
```

--------------------------------

### Selectbox Interaction

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=configuration&slug=layouts

Simulates selecting an option in the `st.selectbox` widget.

```APIDOC
## Selectbox

### Description
A representation of `st.selectbox`.

### Method
Python (AppTest interaction)

### Endpoint
N/A

### Parameters
None

### Request Example
```python
at.selectbox[0].select("New York").run()
```

### Response
N/A

### Response Example
N/A
```

--------------------------------

### Configure Streamlit Secrets for MySQL

Source: https://docs.streamlit.io/develop/tutorials/databases/mysql

TOML configuration for Streamlit's secrets management, specifying MySQL connection details like host, port, database, username, and password. This file should not be committed to version control.

```toml
# .streamlit/secrets.toml

[connections.mysql]
dialect = "mysql"
host = "localhost"
port = 3306
database = "xxx"
username = "xxx"
password = "xxx"
query = { charset = "xxx" }
```

--------------------------------

### Pass Arguments to Streamlit Script

Source: https://docs.streamlit.io/develop/concepts/architecture/run-your-app

How to pass custom arguments to your Streamlit script. Arguments intended for the script must be placed after two dashes (`--`) to distinguish them from Streamlit's own arguments.

```bash
streamlit run your_script.py [-- script args]
```

--------------------------------

### Display HiPlot Experiment

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=configuration&slug=data

Visualizes high-dimensional data interactively using HiPlot. It takes an iterable of dictionaries representing experimental data.

```python
import hiplot as hip

data = [
    {'dropout': 0.1, 'lr': 0.001, 'loss': 10.0, 'optimizer': 'SGD'},
    {'dropout': 0.15, 'lr': 0.01, 'loss': 3.5, 'optimizer': 'Adam'},
    {'dropout': 0.3, 'lr': 0.1, 'loss': 4.5, 'optimizer': 'Adam'}
]
hip.Experiment.from_iterable(data).display()
```

--------------------------------

### Pandas Profiling

Source: https://docs.streamlit.io/develop/api-reference_slug=advanced-features&slug=prerelease

Generates and displays detailed profile reports for pandas DataFrames.

```APIDOC
## Pandas Profiling

### Description
Pandas profiling component for Streamlit.

### Method
N/A (This is a component function)

### Endpoint
N/A

### Parameters
#### Path Parameters
N/A

#### Query Parameters
N/A

#### Request Body
N/A

### Request Example
```python
import pandas as pd
from streamlit_pandas_profiling import st_profile_report

df = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")
pr = df.profile_report()

st_profile_report(pr)
```

### Response
#### Success Response (200)
N/A (Renders a pandas profiling report)

#### Response Example
N/A
```

--------------------------------

### Configure Tableau Secrets in TOML

Source: https://docs.streamlit.io/develop/tutorials/databases/tableau

This TOML snippet configures the necessary credentials for connecting to Tableau Server. It includes the personal access token name, secret, server URL, and site ID. This file should be stored locally in `.streamlit/secrets.toml` and not committed to version control.

```toml
# .streamlit/secrets.toml

[tableau]
token_name = "xxx"
token_secret = "xxx"
server_url = "https://abc01.online.tableau.com/"
site_id = "streamlitexample"  # in your site's URL behind the server_url

```

--------------------------------

### Load Iframe URL

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=configuration&slug=status

This section explains how to load a remote URL in an iframe using the `iframe` function.

```APIDOC
## POST /api/components/iframe

### Description
Loads a remote URL in an iframe.

### Method
POST

### Endpoint
`/api/components/iframe`

### Parameters
#### Query Parameters
- **url** (string) - Required - The URL to load in the iframe.

### Request Example
```python
from st.components.v1 import iframe
iframe(
    "docs.streamlit.io"
)
```

### Response
#### Success Response (200)
- **status** (string) - Indicates the URL was loaded successfully in the iframe.
```

--------------------------------

### Display a number input widget in Python

Source: https://docs.streamlit.io/develop/api-reference/widgets

Allows users to input a numeric value within a specified range. Returns the entered number. Useful for quantitative inputs.

```python
choice = st.number_input("Pick a number", 0, 10)
```

--------------------------------

### Set Default Tab and Style Labels with Markdown in Streamlit

Source: https://docs.streamlit.io/develop/api-reference/layout/st

Illustrates how to set a default active tab using the 'default' parameter and how to style tab labels with GitHub-flavored Markdown, including emojis and text formatting, within Streamlit.

```python
import streamlit as st

tab1, tab2, tab3 = st.tabs(
    [":cat: Cat", ":dog: Dog", ":rainbow[Owl]"], default=":rainbow[Owl]"
)

with tab1:
    st.header("A cat")
    st.image("https://static.streamlit.io/examples/cat.jpg", width=200)
with tab2:
    st.header("A dog")
    st.image("https://static.streamlit.io/examples/dog.jpg", width=200)
with tab3:
    st.header("An owl")
    st.image("https://static.streamlit.io/examples/owl.jpg", width=200)
```

--------------------------------

### Displaying Altair Charts in Streamlit

Source: https://docs.streamlit.io/develop/api-reference_slug=%28&slug=develop&slug=concepts&slug=configuration

Integrates Altair charts into Streamlit applications. Requires an Altair chart object.

```Python
import streamlit as st
import altair as alt

# Assuming my_altair_chart is an Altair chart object
# Example:
# my_altair_chart = alt.Chart(pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})).mark_line().encode(x='x', y='y')

st.altair_chart(my_altair_chart)
```

--------------------------------

### ColorPicker Testing

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

Simulates interactions with the `st.color_picker` element.

```APIDOC
## ColorPicker Testing

### Description
A representation of `st.color_picker`.

### Method
Python

### Endpoint
N/A

### Parameters
#### Path Parameters
N/A

#### Query Parameters
N/A

#### Request Body
N/A

### Request Example
```python
at.color_picker[0].pick("#FF4B4B").run()
```

### Response
#### Success Response (200)
N/A

#### Response Example
N/A
```

--------------------------------

### Configure Streamlit Secrets for Neon Connection

Source: https://docs.streamlit.io/develop/tutorials/databases/neon

TOML format for storing the Neon database connection URL in Streamlit's secrets file. This allows the Streamlit app to securely access the database without hardcoding credentials.

```toml
# .streamlit/secrets.toml

[connections.neon]
url="postgresql://neondb_owner:xxxxxxxxxxxx@ep-adjective-noun-xxxxxxxx.us-east-2.aws.neon.tech/neondb?sslmode=require"
```

--------------------------------

### Sidebar Elements

Source: https://docs.streamlit.io/develop/quick-reference/cheat-sheet

How to add elements to the Streamlit sidebar, either directly or using a 'with' statement.

```APIDOC
## Add Elements to Sidebar

### Description
Add widgets and other elements to the Streamlit sidebar.

### Method
Python

### Endpoint
N/A

### Parameters
None

### Request Example
```python
# Just add it after st.sidebar:
a = st.sidebar.radio("Select one:", [1, 2])

# Or use "with" notation:
with st.sidebar:
    st.radio("Select one:", [1, 2])
```

### Response
N/A
```

--------------------------------

### Display Video Player in Streamlit

Source: https://docs.streamlit.io/develop/api-reference/media

Shows how to embed a video player in a Streamlit app, accepting inputs such as NumPy arrays, video bytes, file objects, and URLs. Enables in-app video playback.

```python
st.video(numpy_array)
st.video(video_bytes)
st.video(file)
st.video("https://example.com/myvideo.mp4", format="video/mp4")
```

--------------------------------

### Selectbox Testing

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=api-reference&slug=testing&slug=st.testing.v1.apptest

Simulates interactions with the `st.selectbox` element.

```APIDOC
## Selectbox Testing

### Description
A representation of `st.selectbox`.

### Method
Python

### Endpoint
N/A

### Parameters
#### Path Parameters
N/A

#### Query Parameters
N/A

#### Request Body
N/A

### Request Example
```python
at.selectbox[0].select("New York").run()
```

### Response
#### Success Response (200)
N/A

#### Response Example
N/A
```

--------------------------------

### Display Download Button Widget

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=configuration&slug=widgets

This code displays a download button widget in a Streamlit application. It takes a file as input, allowing users to download the specified file. This is useful for providing downloadable content.

```Python
import streamlit as st
st.download_button("Download file", file)
```

--------------------------------

### Pandas Profiling Report in Streamlit

Source: https://docs.streamlit.io/develop/api-reference/data

Integrates Pandas Profiling reports directly into Streamlit apps, providing comprehensive data exploration and analysis. Requires `streamlit-pandas-profiling`.

```python
import pandas as pd
from streamlit_pandas_profiling import st_profile_report

df = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")
pr = df.profile_report()

st_profile_report(pr)
```

--------------------------------

### Streamlit Navigation and Pages

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=configuration&slug=widgets

APIs for configuring navigation, defining pages, and linking between pages in a multipage Streamlit application.

```APIDOC
## Navigation and Pages

### Description
APIs for configuring navigation, defining pages, and linking between pages in a multipage Streamlit application.

### Navigation

#### Method
`st.navigation(pages)`

#### Description
Configures the available pages in a multipage app.

#### Parameters
- **pages** (dict) - Required - A dictionary mapping page names to page configurations.

### Page

#### Method
`st.Page(script_path, title, icon)`

#### Description
Defines a page in a multipage app.

#### Parameters
- **script_path** (str) - Required - The path to the page's script.
- **title** (str) - Required - The title of the page.
- **icon** (str) - Optional - The icon for the page (e.g., ':material/home:').

### Page Link

#### Method
`st.page_link(page, label, icon)`

#### Description
Displays a link to another page in a multipage app.

#### Parameters
- **page** (str) - Required - The path to the target page.
- **label** (str) - Required - The text label for the link.
- **icon** (str) - Optional - The icon for the link.

### Switch Page

#### Method
`st.switch_page(page)`

#### Description
Programmatically navigates to a specified page.

#### Parameters
- **page** (str) - Required - The path to the page to switch to.
```

--------------------------------

### Display Vega-Lite Charts in Streamlit

Source: https://docs.streamlit.io/develop/api-reference_slug=develop&slug=concepts&slug=configuration&slug=charts

Renders charts defined using the Vega-Lite grammar within a Streamlit application. Requires a Vega-Lite chart specification object.

```python
import streamlit as st

# Assuming my_vega_lite_chart is a Vega-Lite chart specification object
# st.vega_lite_chart(my_vega_lite_chart)
```

--------------------------------

### Set Streamlit Client Option using Python

Source: https://docs.streamlit.io/develop/api-reference/configuration/st

This snippet demonstrates how to use the `st.set_option` function in Python to configure Streamlit applications. It specifically shows how to enable error details display. Note that changing options may require an app rerun to take effect.

```python
import streamlit as st

st.set_option("client.showErrorDetails", True)
```

--------------------------------

### Create Streamlit Logout Page Function

Source: https://docs.streamlit.io/develop/tutorials/multipage/dynamic-navigation

This Python code defines a `logout` function for a Streamlit application. Upon execution, it immediately sets the user's role in `st.session_state` to `None` and reruns the application. This effectively logs the user out and redirects them, likely to a login page, without rendering any visible elements on the logout page itself.

```python
def logout():
    st.session_state.role = None
    st.rerun()
```

--------------------------------

### Configure Streamlit Secrets for TiDB (TOML)

Source: https://docs.streamlit.io/develop/tutorials/databases/tidb

TOML configuration for Streamlit's secrets management, specifying connection details for a TiDB database. This file should be placed in `.streamlit/secrets.toml` and includes host, port, database, username, and password. It's crucial for secure credential handling.

```toml
# .streamlit/secrets.toml

[connections.tidb]
dialect = "mysql"
host = "<TiDB_cluster_host>"
port = 4000
database = "pets"
username = "<TiDB_cluster_user>"
password = "<TiDB_cluster_password>"
```

--------------------------------

### Enable Data Caching with @st.cache_data in Python

Source: https://docs.streamlit.io/get-started/tutorials/create-an-app

This snippet demonstrates how to use the @st.cache_data decorator in Python to enable caching for a data loading function. Caching stores function results based on input parameters and function code, improving performance by avoiding re-computation on subsequent calls with the same inputs. This is particularly useful for long-running data loading operations.

```python
@st.cache_data
def load_data(nrows):

```

```python
data_load_state.text("Done! (using st.cache_data)")

```

--------------------------------

### Create Page Links in Streamlit

Source: https://docs.streamlit.io/develop/api-reference/navigation

Generates clickable links within a Streamlit app that navigate to other pages. You can specify the target page file, a display label, and an icon for the link. This is useful for creating navigation menus or direct links to specific sections.

```python
st.page_link("app.py", label="Home", icon="🏠")
st.page_link("pages/profile.py", label="Profile")
```

--------------------------------

### Streamlit Conditional Navigation with st.navigation

Source: https://docs.streamlit.io/develop/concepts/multipage-apps/page-and-navigation

This Python code demonstrates how to create a dynamic navigation menu in Streamlit that changes based on user login status. It uses st.session_state to track login status and st.navigation to display different sets of pages for logged-in and logged-out users. The code defines pages for login, logout, dashboard, bug reports, system alerts, search, and history.

```python
import streamlit as st

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    if st.button("Log in"):
        st.session_state.logged_in = True
        st.rerun()

def logout():
    if st.button("Log out"):
        st.session_state.logged_in = False
        st.rerun()

login_page = st.Page(login, title="Log in", icon=":material/login:")
logout_page = st.Page(logout, title="Log out", icon=":material/logout:")

dashboard = st.Page(
    "reports/dashboard.py", title="Dashboard", icon=":material/dashboard:", default=True
)
bugs = st.Page("reports/bugs.py", title="Bug reports", icon=":material/bug_report:")
alerts = st.Page(
    "reports/alerts.py", title="System alerts", icon=":material/notification_important:"
)

search = st.Page("tools/search.py", title="Search", icon=":material/search:")
history = st.Page("tools/history.py", title="History", icon=":material/history:")

if st.session_state.logged_in:
    pg = st.navigation(
        {
            "Account": [logout_page],
            "Reports": [dashboard, bugs, alerts],
            "Tools": [search, history],
        }
    )
else:
    pg = st.navigation([login_page])

pg.run()
```