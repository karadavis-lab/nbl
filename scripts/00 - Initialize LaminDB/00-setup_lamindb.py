import lamindb as ln
import typer

import nbl

logger = nbl.logger
app = typer.Typer()


@app.command(no_args_is_help=True)
def setup_lamindb(
    project_name: str = typer.Argument(..., help="Name of the project", allow_dash=True),
    abbreviation: str = typer.Argument(..., help="Abbreviation of the project"),
):
    """Setup a LaminDB project.

    Args:
        project_name
            Name of the project.
        abbreviation
            Abbreviation of the project.
    """
    typer.echo(f"Setting up LaminDB project {project_name}...")
    logger.info("Setting up LaminDB")
    ln.Project(name=project_name, abbr=abbreviation).save()
    ln.track(project=project_name)
    ln.finish()
    logger.info("LaminDB setup complete")


if __name__ == "__main__":
    app()
