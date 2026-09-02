from data.sim.generation.tasks._pick import create_scripted_pick


def create(context):
    return create_scripted_pick(context, generator_id="scripted_pick")
