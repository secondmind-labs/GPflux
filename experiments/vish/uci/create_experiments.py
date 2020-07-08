import os
from itertools import product


class CommandsBuilder:
    r"""
    Creates the outer-product of configurations to be executed.
    Returns a list with all the combinations.

    Here's an example:
    ```
    commands = (
        CommandsBuilder()
        .add("dataset", ["Power", "Kin8mn"])
        .add("split", [0, 1])
        .build()
    )
    ```
    Returns
    ```
    commands = [
        "python main.py with --dataset=Power --split=0;",
        "python main.py with --dataset=Power --split=1;",
        "python main.py with --dataset=Kin8mn --split=0;",
        "python main.py with --dataset=Kin8mn --split=1;",
    ]
    ```
    """
    command_template = "python main.py -p with{config};"
    single_config_template = " {key}={value}"

    keys = []
    values = []

    def add(self, key, values):
        self.keys.append(key)
        self.values.append(values)
        return self

    def build(self):
        commands = []
        for args in product(*self.values):
            config = ""
            for key, value in zip(self.keys, args):
                config += self.single_config_template.format(key=key, value=value)
            command = self.command_template.format(config=config)
            commands.append(command)
        return commands


DATASETS_WILSON = [
    "Wilson_3droad",
    "Wilson_houseelectric",
    "Wilson_protein",
    "Wilson_slice",
    "Wilson_kin40k",
]

DATASETS_UCI = ["Power", "Energy", "Concrete", "Kin8mn", "Yacht"]

DATASETS_AIRLINE = ["airline10k", "airline100k", "airline1m"]


if __name__ == "__main__":
    NAME = "commands_uci.txt"

    if os.path.exists(NAME):
        print("File to store script already exists", NAME)
        exit(-1)

    commands = (
        CommandsBuilder()
        .add("model_type", ["vish"])
        .add("dataset", DATASETS_UCI)
        .add("split", range(5))
        .build()
    )

    with open(NAME, "w") as file:
        file.write("\n".join(commands))
