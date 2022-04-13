SECTIONS
{
    .__lisa_features : {
        __lisa_features_start = . ;
        KEEP(*(.__lisa_features))
        __lisa_features_stop = . ;
    }
}

