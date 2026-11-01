"""
App metadata for the TimeFrame Winnower, kept free of heavy imports.

DO NOT CHANGE the name of the file
"""

from mmif import DocumentTypes, AnnotationTypes

from clams.app import ClamsApp
from clams.appmetadata import AppMetadata


# DO NOT CHANGE the function name
def appmetadata() -> AppMetadata:
    """
    Set app-metadata values and return them as an ``AppMetadata`` object.
    Read these documentations before changing the code below
    - https://clams.ai/clams-python/appmetadata.html metadata specification.
    - https://clams.ai/clams-python/autodoc/clams.appmetadata.html python API

    :return: AppMetadata object holding all necessary information.
    """

    metadata = AppMetadata(
        name="TimeFrame Winnower",
        description=(
            "Selects segment boundaries by agreement across upstream views. "
            "Every view in the input MMIF that carries time-anchored "
            "annotations is one source of candidate boundary events: the "
            "endpoints of its Sentence annotations when it has any, otherwise "
            "the endpoints of its TimeFrame annotations (word-level frames "
            "aligned to Tokens are skipped). Events are widened into a window "
            "of `windowMs`, and a boundary is a point where at least "
            "`minAgreement` distinct sources agree. Among boundaries closer "
            "than `minSeparationMs`, the best-corroborated set is selected "
            "exactly by dynamic programming. Output is one `segment` "
            "TimeFrame per interval between consecutive boundaries. The "
            "method needs no model and no GPU. The default parameters are "
            "the ones used over five sources (whisper-wrapper, "
            "pyannotesd-wrapper, pyscenedetect-wrapper, "
            "inaspeechsegmenter-wrapper, swt-detection) in the JCDL 2026 "
            "paper (https://doi.org/10.1145/3805696.3846037)."
        ),
        app_license="Apache 2.0",
        identifier="timeframe-winnower",
        url="https://github.com/clamsproject/app-timeframe-winnower",
        est_gpu_mem_min=0,
        est_gpu_mem_typ=0,
    )

    metadata.add_input(DocumentTypes.VideoDocument)
    in_tf = metadata.add_input(AnnotationTypes.TimeFrame, required=False)
    in_tf.add_description(
        'Time-anchored annotations from any upstream app. Each view that '
        'contains them is one voting source. Views must declare `timeUnit`. '
        'Filter by label with `tfLabels` and `tfLabelIgnore`.')
    in_sent = metadata.add_input(AnnotationTypes.Sentence, required=False)
    in_sent.add_description(
        'Sentence annotations whose `targets` are Tokens aligned to '
        'TimeFrames (as produced by ASR apps such as whisper-wrapper). When a '
        'view has Sentences, their endpoints are used as that view\'s events '
        'instead of its TimeFrames.')
    metadata.add_input(AnnotationTypes.Token, required=False)
    metadata.add_input(AnnotationTypes.Alignment, required=False)

    out_tf = metadata.add_output(AnnotationTypes.TimeFrame,
                                 timeUnit='milliseconds', label='segment')
    out_tf.add_description(
        'One TimeFrame per segment between consecutive selected boundaries, '
        'tiling the whole recording. Start and end are integer milliseconds.')

    metadata.add_parameter(
        name='windowMs', type='integer', default=5000,
        description='W: width in milliseconds of the window each candidate '
                    'event is widened into before sources are compared.')
    metadata.add_parameter(
        name='minAgreement', type='integer', default=4,
        description='K: minimum number of distinct source views that must '
                    'agree for a point to become a candidate boundary. The '
                    'default suits five sources (all but one must agree). '
                    'Lower it when fewer views are present, or no boundary '
                    'can be selected.')
    metadata.add_parameter(
        name='minSeparationMs', type='integer', default=20000,
        description='M: minimum distance in milliseconds between two '
                    'selected boundaries, and from the start and the end of '
                    'the recording. Encodes the shortest admissible segment. '
                    '0 disables the constraint.')
    metadata.add_parameter(
        name='tfLabels', type='string', default=[], multivalued=True,
        description='Label(s) of input TimeFrame annotations to use as '
                    'candidate events. By default (`[]`) every TimeFrame is '
                    'used regardless of label. When set, TimeFrames without a '
                    'label are not used. Sentence-derived events are not '
                    'affected.')
    metadata.add_parameter(
        name='tfLabelIgnore', type='string', default=[], multivalued=True,
        description='Label(s) of input TimeFrame annotations to leave out, '
                    'applied after `tfLabels`. TimeFrames without a label are '
                    'never dropped by this list. Sentence-derived events are '
                    'not affected.')

    return metadata


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(metadata.jsonify(pretty=True))
