"""A class to help with writting markdown."""


class DocsyTools:  # TODO All abstract methods should be implemented
    """ """

    def __init__(self):
        return  # TODO Super __init__ should be called at some point

    @classmethod
    def write_class_to_markdown(
        cls, class_to_write, fn_out, method_to_omit=[], omit_private_methods=True, omit_parent_methods=True
    ):

        methods_to_write = cls._get_list_of_methods(class_to_write)

        for method in methods_to_write:
            method_block = cls._method_to_str(getattr(class_to_write, method))

    @classmethod
    def _method_to_str(cls, method_name):
        pass

    @classmethod
    def _get_list_of_methods(
        cls, class_to_search, methods_to_omit=[], omit_private_methods=True, omit_parent_methods=True
    ):
        """
        Returns a list of methods inside a provided COAsT class, with some
        other options

        Parameters
        ----------
        class_to_search : imported class
            Class imported from COAsT (e.g. from coast import Profile)
        methods_to_omit : list
            List of method strings to omit from the output. The default is [].
        omit_private_methods : bool, optional
            If true, omit methods beginning with "_". The default is True.
        omit_parent_methods : bool, optional
           If true, omit methods in any parent/ancestor class. The default is True.

        Returns
        -------
        methods_to_write : list
            List of strings denoting method names

        """

        # Get list of methods
        methods_to_write = dir(class_to_search)

        if omit_parent_methods:
            # Get parent class and methods (if user wants)
            parents = class_to_search.__mro__[1:-1]
            parent_methods = []

            # Get (non-unique) parent methods
            for pp in parents:
                methods_pp = dir(pp)
                [parent_methods.append(mm) for mm in methods_pp]

            # Remove classes from methods list if they are the same name
            for method in parent_methods:
                if method in methods_to_write:
                    methods_to_write.remove(method)

        # Look for methods beginning with "_" and remove them (if true)
        if omit_private_methods:
            for method in methods_to_write:
                if method[0] == "_":
                    methods_to_write.remove(method)

        return methods_to_write
