import beelabel.alignment


def test_grab_photos_in_timerange():
    # TODO generate empty files in a temp dir

    beelabel.alignment.grab_photos_in_timerange(
        path='photos/system001',
        starttime='08:00:00',
        endtime='08:10:00'
    )

    # TODO delete temp files
