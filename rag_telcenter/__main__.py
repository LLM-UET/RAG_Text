import click

@click.group()
def cli():
    pass

@cli.command()
def demo():
    """Run the demo realization."""
    from .realizations.demo import main
    main()

@cli.command()
def serve():
    """Run the server realization."""
    from .realizations.server import main
    main()

if __name__ == "__main__":
    cli()
