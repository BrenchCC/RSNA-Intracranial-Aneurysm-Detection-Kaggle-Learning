"""Shared constants and lightweight helpers for mock RSNA pipeline."""

LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present'
]

VESSEL_CLASS_TO_LABEL = {
    'L_ICA_INFRA': 'Left Infraclinoid Internal Carotid Artery',
    'R_ICA_INFRA': 'Right Infraclinoid Internal Carotid Artery',
    'L_ICA_SUPRA': 'Left Supraclinoid Internal Carotid Artery',
    'R_ICA_SUPRA': 'Right Supraclinoid Internal Carotid Artery',
    'L_MCA': 'Left Middle Cerebral Artery',
    'R_MCA': 'Right Middle Cerebral Artery',
    'A_COM': 'Anterior Communicating Artery',
    'L_ACA': 'Left Anterior Cerebral Artery',
    'R_ACA': 'Right Anterior Cerebral Artery',
    'L_PCOM': 'Left Posterior Communicating Artery',
    'R_PCOM': 'Right Posterior Communicating Artery',
    'BASILAR_TIP': 'Basilar Tip',
    'OTHER_POST': 'Other Posterior Circulation'
}

VESSEL_CLASSES = list(VESSEL_CLASS_TO_LABEL.keys())


def clamp_prob(value):
    """Clamp a numeric value into [0, 1].

    Args:
        value (float): Input value.

    Returns:
        float: Clamped value.
    """
    return float(max(0.0, min(1.0, value)))


def sigmoid(x):
    """Compute numerically stable sigmoid.

    Args:
        x (float): Input scalar.

    Returns:
        float: Sigmoid result.
    """
    if x >= 0:
        z = 1.0 / (1.0 + pow(2.718281828459045, -x))
        return float(z)

    exp_x = pow(2.718281828459045, x)
    return float(exp_x / (1.0 + exp_x))
