"""Provider fields that may be returned to a patient."""

# Catalogue specialty -> the label the directory stores for it. The provider
# search analyzer adds the label as a synonym alternative, so a query for
# "ophthalmology" also reaches a document whose specialty is "eye doctor".
SPECIALTY_LABELS = {
    "gynecology": "ob gyn",
    "ophthalmology": "eye doctor",
}

PROVIDER_PUBLIC_FIELDS = [
    "provider_id",
    "name",
    "specialty",
    "subspecialties",
    "languages",
    "city",
    "rating",
    "review_count",
    "accepting_new_patients",
    "next_available",
]
