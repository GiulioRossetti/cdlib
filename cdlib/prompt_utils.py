def report_missing_packages(package_list: list):
    if len(package_list) > 0:
        print(
            "Note: to be able to use all crisp methods, you need to install some additional packages: ",
            package_list,
        )


def prompt_import_failure(package: str, exception: Exception, show_details=False):
    print("Note: cannot import package: %s" % str(package))
    if show_details:
        print("------> Detailed exception:")
        print(exception)
