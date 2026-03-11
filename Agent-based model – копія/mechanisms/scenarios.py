"""
Scenario configuration and gradual transitions.
"""
from typing import Dict
import yaml
from pathlib import Path


def get_scenario_config(scenario_mode: str) -> Dict:
    """
    Load scenario configuration.

    Args:
        scenario_mode: Scenario name

    Returns:
        Configuration dictionary
    """
    config_dir = Path(__file__).parent.parent / 'config'

    scenario_map = {
        'BASELINE': 'baseline.yaml',
        'AED_PILLAR1': 'aed_pillar1.yaml',
        'AED_FULL': 'aed_full.yaml'
    }

    filename = scenario_map.get(scenario_mode, 'baseline.yaml')
    filepath = config_dir / filename

    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def apply_gradual_transition(config: Dict, current_period: int) -> Dict:
    """
    Apply gradual parameter changes for AED_GRADUAL scenario.

    Transition Phases:
    - Steps 0-4: BASELINE
    - Steps 5-9: AED_PILLAR1 (debt restructuring enabled)
    - Steps 10-14: AED_FULL with reduced tax (15%)
    - Steps 15+: AED_FULL with zero tax

    Args:
        config: Current configuration dict
        current_period: Current simulation step

    Returns:
        Updated config dict
    """
    # Mesa 3.5.0: model.steps starts at 1 (incremented before step() runs)
    # Phase boundaries are adjusted so that:
    #   After run_for(5):  steps 1-5  -> BASELINE
    #   After run_for(10): steps 6-10 -> AED_PILLAR1
    #   After run_for(15): steps 11-15 -> AED_FULL @15% tax
    #   After run_for(20): steps 16+  -> AED_FULL @0% tax
    phases = {
        'phase_0': {'start': 0, 'end': 6, 'mode': 'BASELINE'},
        'phase_1': {'start': 6, 'end': 11, 'mode': 'AED_PILLAR1'},
        'phase_2': {'start': 11, 'end': 16, 'mode': 'AED_FULL', 'tax_rate': 0.15},
        'phase_3': {'start': 16, 'end': None, 'mode': 'AED_FULL', 'tax_rate': 0.0}
    }

    for phase_name, phase_config in sorted(phases.items()):
        start = phase_config['start']
        end = phase_config.get('end')

        if current_period >= start and (end is None or current_period < end):
            # Apply this phase's parameters
            phase_scenario = get_scenario_config(phase_config['mode'])
            config.update(phase_scenario)

            # Apply phase-specific overrides
            for key, value in phase_config.items():
                if key not in ('start', 'end', 'mode'):
                    config[key] = value

            break

    return config
